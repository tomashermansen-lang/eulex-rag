from src.tools.distances import DistanceRow, summarize_best_distances, sweep_thresholds


def test_summarize_best_distances_empty():
    summary = summarize_best_distances([])
    assert summary["count"] == 0
    assert summary["min"] is None


def test_summarize_best_distances_quantiles():
    rows = [
        DistanceRow(id="a", question="q", best_distance=0.1, distances=[0.1], top_sources=[]),
        DistanceRow(id="b", question="q", best_distance=0.5, distances=[0.5], top_sources=[]),
        DistanceRow(id="c", question="q", best_distance=0.9, distances=[0.9], top_sources=[]),
    ]

    summary = summarize_best_distances(rows)
    assert summary["count"] == 3
    assert summary["min"] == 0.1
    assert summary["max"] == 0.9
    assert summary["p50"] == 0.5


def test_sweep_thresholds_counts():
    rows = [
        DistanceRow(id="a", question="q", best_distance=0.10, distances=[0.10], top_sources=[]),
        DistanceRow(id="b", question="q", best_distance=0.50, distances=[0.50], top_sources=[]),
        DistanceRow(id="c", question="q", best_distance=None, distances=[], top_sources=[]),
    ]

    sweep = sweep_thresholds(rows, start=0.10, end=0.50, step=0.20)
    assert [round(r["threshold"], 2) for r in sweep] == [0.10, 0.30, 0.50]

    # threshold=0.10 => abstain for 0.50 and None => 2
    assert sweep[0]["abstain"] == 2
    assert sweep[0]["newly_answerable_ids"] == ["a"]
    # threshold=0.50 => abstain only for None => 1
    assert sweep[-1]["abstain"] == 1
    assert sweep[-1]["newly_answerable_ids"] == ["b"]
