"""AI-powered metrics analysis engine module.

Single Responsibility: Construct analysis prompt from a metrics snapshot
and stream LLM tokens via the shared llm_client.

Does NOT import rag.py or any other orchestrator.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator

from ..common.config_loader import get_settings_yaml
from ..engine.llm_client import call_llm_stream_async


def _build_analysis_prompt(snapshot: dict[str, Any]) -> str:
    """Build a structured analysis prompt from a metrics snapshot.

    Args:
        snapshot: Dict with overview, quality, performance, ingestion data.

    Returns:
        Prompt string for the LLM.
    """
    sections = []

    # ── Overview ──
    trend = snapshot.get("trend", {})
    trend_dir = trend.get("direction", "unknown")
    trend_delta = trend.get("delta_pp")
    trend_str = f" (trend: {trend_dir}"
    if trend_delta is not None:
        trend_str += f", {trend_delta:+.1f}pp"
    trend_str += ")"

    sections.append(
        f"Samlet bestået-rate: {snapshot.get('unified_pass_rate', 0)}%{trend_str}\n"
        f"Sundhedsstatus: {snapshot.get('health_status', 'unknown')}\n"
        f"Single-law: {snapshot.get('single_law', {}).get('passed', 0)}"
        f"/{snapshot.get('single_law', {}).get('total', 0)}"
        f" ({snapshot.get('single_law', {}).get('pass_rate', 0)}%)\n"
        f"Cross-law: {snapshot.get('cross_law', {}).get('passed', 0)}"
        f"/{snapshot.get('cross_law', {}).get('total', 0)}"
        f" ({snapshot.get('cross_law', {}).get('pass_rate', 0)}%)"
    )

    # ── Quality: per-law pass rates ──
    per_law = snapshot.get("per_law", [])
    if per_law:
        law_lines = [
            f"  - {l['law']}: {l['pass_rate']}% ({l['passed']}/{l['total']})"
            for l in sorted(per_law, key=lambda x: x["pass_rate"])
        ]
        sections.append("Bestået-rate per lov (sorteret, laveste først):\n"
                         + "\n".join(law_lines))

    # ── Quality: per-mode ──
    per_mode = snapshot.get("per_mode", [])
    if per_mode:
        mode_lines = [
            f"  - {m['mode']}: {m['pass_rate']}% ({m['passed']}/{m['total']})"
            for m in per_mode
        ]
        sections.append("Per tilstand (synthesis mode):\n" + "\n".join(mode_lines))

    # ── Quality: per-difficulty ──
    per_diff = snapshot.get("per_difficulty", [])
    if per_diff:
        diff_lines = [
            f"  - {d['difficulty']}: {d['pass_rate']}% ({d['passed']}/{d['total']})"
            for d in per_diff
        ]
        sections.append("Per sværhedsgrad:\n" + "\n".join(diff_lines))

    # ── Quality: per-scorer ──
    per_scorer = snapshot.get("per_scorer", [])
    if per_scorer:
        scorer_lines = [
            f"  - {s['scorer']}: {s['pass_rate']}% ({s['passed']}/{s['total']})"
            for s in per_scorer
        ]
        sections.append("Per scorer:\n" + "\n".join(scorer_lines))

    # ── Failing cases with details ──
    failing = snapshot.get("failing_cases", [])
    if failing:
        fail_lines = []
        for fc in failing:
            scorers_str = ", ".join(
                f"{s}: {msg}" for s, msg in fc.get("failed_scorers", {}).items()
            )
            fail_lines.append(
                f"  - [{fc.get('law', '?')}] {fc.get('case_id', '?')}: {scorers_str}"
            )
        sections.append("Fejlende cases (detaljer):\n" + "\n".join(fail_lines))

    # ── Performance: overall latency ──
    percentiles = snapshot.get("percentiles", {})
    if percentiles:
        sections.append(
            f"Samlet latency: P50={percentiles.get('p50', 0):.2f}s, "
            f"P95={percentiles.get('p95', 0):.2f}s, "
            f"P99={percentiles.get('p99', 0):.2f}s"
        )

    # ── Performance: SL vs CL latency ──
    sl_pct = snapshot.get("sl_percentiles", {})
    cl_pct = snapshot.get("cl_percentiles", {})
    if sl_pct or cl_pct:
        lat_lines = []
        if sl_pct:
            lat_lines.append(
                f"  - Single-law: P50={sl_pct.get('p50', 0):.2f}s, "
                f"P95={sl_pct.get('p95', 0):.2f}s"
            )
        if cl_pct:
            lat_lines.append(
                f"  - Cross-law: P50={cl_pct.get('p50', 0):.2f}s, "
                f"P95={cl_pct.get('p95', 0):.2f}s"
            )
        sections.append("Latency per evaltype:\n" + "\n".join(lat_lines))

    # ── Performance: per-run-mode latency (SL) ──
    run_mode_lat = snapshot.get("run_mode_latency", [])
    if run_mode_lat:
        rm_lines = [
            f"  - {m['run_mode']}: P50={m['p50_seconds']:.2f}s, "
            f"P95={m['p95_seconds']:.2f}s ({m['case_count']} cases)"
            for m in run_mode_lat
        ]
        sections.append("Latency pr. run mode (SL):\n" + "\n".join(rm_lines))

    # ── Performance: escalation & retry ──
    esc_rate = snapshot.get("escalation_rate")
    retry_rate = snapshot.get("retry_rate")
    if esc_rate is not None or retry_rate is not None:
        stab_parts = []
        if esc_rate is not None:
            stab_parts.append(f"Eskaleringsrate: {esc_rate}%")
        if retry_rate is not None:
            stab_parts.append(f"Genforsøgsrate: {retry_rate}%")
        sections.append("Stabilitet (SL):\n  " + "\n  ".join(stab_parts))

    # ── Performance: escalation & retry trends ──
    esc_trend = snapshot.get("escalation_trend", [])
    retry_trend = snapshot.get("retry_trend", [])
    if esc_trend or retry_trend:
        trend_lines = []
        if len(esc_trend) >= 2:
            first_esc = esc_trend[0]["rate"]
            last_esc = esc_trend[-1]["rate"]
            trend_lines.append(
                f"  - Eskalering: {first_esc}% → {last_esc}% "
                f"(seneste {len(esc_trend)} dage)"
            )
        if len(retry_trend) >= 2:
            first_ret = retry_trend[0]["rate"]
            last_ret = retry_trend[-1]["rate"]
            trend_lines.append(
                f"  - Genforsøg: {first_ret}% → {last_ret}% "
                f"(seneste {len(retry_trend)} dage)"
            )
        if trend_lines:
            sections.append("Trends (stabilitet):\n" + "\n".join(trend_lines))

    # ── HTML-tjek (structure coverage, N/A excluded) ──
    html_overall = snapshot.get("html_tjek_overall")
    html_checked = snapshot.get("html_tjek_checked", 0)
    html_na = snapshot.get("html_tjek_na", 0)
    html_corpora = snapshot.get("html_tjek_corpora", [])
    if html_overall is not None:
        ing_lines = [
            f"HTML-tjek: {html_overall}% "
            f"({html_checked} love tjekket, {html_na} ikke tjekket endnu)"
        ]
        for c in html_corpora:
            pct = c.get("html_tjek_pct")
            if pct is not None:
                ing_lines.append(
                    f"  - {c['name']}: {pct}% "
                    f"({c.get('chunks', 0)} chunks, "
                    f"{c.get('unhandled', 0)} ubehandlede)"
                )
            else:
                ing_lines.append(f"  - {c['name']}: ikke tjekket")
        sections.append("\n".join(ing_lines))

    metrics_block = "\n\n".join(sections)

    return (
        "Du er en senior kvalitetsanalytiker for et RAG-system til EU-lovgivning.\n\n"
        "VIGTIG REGEL: Du må IKKE gentage aggregerede tal som opmærksomhedspunkter. "
        "Brugeren kan allerede se tallene i dashboardet. "
        "Din værdi er at analysere de FEJLENDE CASES i bunden af data og finde "
        "mønstre, årsager og sammenhænge som tallene alene ikke viser.\n\n"
        f"--- METRICS ---\n{metrics_block}\n--- END METRICS ---\n\n"
        "Svar på dansk med velformateret markdown. "
        "Brug nummereret liste for opmærksomhedspunkter og anbefalinger. "
        "Hold afsnit korte (maks 3 sætninger). Adskil hvert punkt tydeligt.\n\n"
        "## SUNDHEDSSTATUS\n"
        "1–2 sætninger. Hvad er den vigtigste ændring siden sidst, og hvad driver den?\n\n"
        "## OPMÆRKSOMHEDSPUNKTER\n"
        "Brug nummereret liste (1. 2. 3. osv.) med 3–5 punkter. "
        "Hvert punkt får en **kort fed overskrift** efterfulgt af 2–3 sætninger. "
        "Hvert punkt SKAL:\n"
        "- Nævne specifikke case-id'er fra fejlende cases\n"
        "- Identificere det underliggende mønster (f.eks. 'alle fejl i cross-law "
        "skyldes missing comparison targets' eller 'faithfulness-fejl koncentreret i én lov')\n"
        "- Foreslå en sandsynlig rodårsag (chunking, routing, corpus-mapping, prompt, etc.)\n\n"
        "Eksempel på godt punkt:\n"
        "> 1. **Manglende anchors i cross-law**: Case auto_inverted_scope og "
        "data_protection_and_trade fejler begge med missing_any_of anchors — "
        "routing finder de rigtige love, men retrieval henter ikke de specifikke artikler. "
        "Sandsynlig årsag: embeddings for generelle artikler (art. 1-2) scorer for lavt.\n\n"
        "Eksempel på dårligt punkt:\n"
        "> 'Cross-law performance er 85%, hvilket er lavere end single-law.' "
        "(dette gentager bare et tal)\n\n"
        "## ANBEFALINGER\n"
        "Brug nummereret liste (1. 2. 3. osv.) med 3–5 forslag. "
        "Hvert forslag får en **kort fed overskrift** efterfulgt af:\n"
        "- Hvilken observation det adresserer\n"
        "- Det konkrete fix (ikke 'forbedre performance')\n"
        "- Forventet effekt: hvilke cases der vil blive grønne\n\n"
        "## ORDLISTE\n"
        "Forklar KUN de tekniske termer og scorer-navne du faktisk brugte i analysen ovenfor. "
        "Maks 6–8 termer. Format: **term** — forklaring i én sætning. "
        "Inkludér ikke generiske termer som brugeren allerede kender (P50, P95, pass rate)."
    )


async def analyse_metrics_stream(
    metrics_snapshot: dict[str, Any],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    """Build prompt from metrics snapshot, stream LLM analysis.

    Args:
        metrics_snapshot: Dict with overview, quality, performance data.
        model: LLM model override (default from config).

    Yields:
        Text chunks from the LLM response.

    Raises:
        RAGEngineError: On LLM failure (network, rate limit, timeout).
    """
    if model is None:
        config = get_settings_yaml().get("dashboard", {})
        model = config.get("ai_analysis_model", "gpt-4o-mini")

    prompt = _build_analysis_prompt(metrics_snapshot)

    async for chunk in call_llm_stream_async(prompt, model=model):
        yield chunk
