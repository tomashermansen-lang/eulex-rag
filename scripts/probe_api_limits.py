#!/usr/bin/env python3
"""Probe OpenAI API rate limits for the eval judge model.

Fires batches of concurrent requests and reports when 429s appear.
Uses the same client/model config as the actual eval scorers.
"""
import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

# Minimal prompt to keep token usage low
PROBE_PROMPT = "Reply with exactly one word: OK"


async def single_call(client: AsyncOpenAI, model: str, call_id: int) -> dict:
    """Make one API call, return timing and status."""
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=5,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - t0
        return {"id": call_id, "status": "ok", "elapsed_ms": round(elapsed * 1000)}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        status = "rate_limited" if "429" in str(e) else f"error: {e}"
        return {"id": call_id, "status": status, "elapsed_ms": round(elapsed * 1000)}


async def probe_batch(client: AsyncOpenAI, model: str, concurrency: int) -> list[dict]:
    """Fire `concurrency` calls simultaneously."""
    tasks = [single_call(client, model, i) for i in range(concurrency)]
    return await asyncio.gather(*tasks)


async def main():
    # Load judge model from config (same as eval uses)
    from src.common.config_loader import get_settings_yaml
    settings = get_settings_yaml()
    judge_model = (
        os.getenv("EVAL_JUDGE_MODEL")
        or settings.get("eval", {}).get("llm_judge", {}).get("model", "gpt-4o-mini")
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key)

    print(f"Model: {judge_model}")
    print(f"{'Concurrency':>12} | {'OK':>4} | {'429s':>4} | {'Errors':>6} | {'Avg ms':>7} | {'Max ms':>7}")
    print("-" * 65)

    # Test increasing concurrency: 5, 10, 15, 20, 25, 30, 40, 50
    for n in [5, 10, 15, 20, 25, 30, 40, 50]:
        results = await probe_batch(client, judge_model, n)

        ok = sum(1 for r in results if r["status"] == "ok")
        rate_limited = sum(1 for r in results if r["status"] == "rate_limited")
        errors = sum(1 for r in results if r["status"] not in ("ok", "rate_limited"))
        avg_ms = round(sum(r["elapsed_ms"] for r in results) / len(results))
        max_ms = max(r["elapsed_ms"] for r in results)

        print(f"{n:>12} | {ok:>4} | {rate_limited:>4} | {errors:>6} | {avg_ms:>7} | {max_ms:>7}")

        if rate_limited > 0:
            print(f"  ^ Hit rate limit at concurrency={n}")

        # Brief pause between batches to not poison the next measurement
        await asyncio.sleep(2)

    print("\nDone. Use these numbers to set eval.scorer_concurrency safely.")


if __name__ == "__main__":
    asyncio.run(main())
