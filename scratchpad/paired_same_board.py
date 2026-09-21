"""Paired Agent_A vs Agent_B comparison on the SAME board (evaluation 3).

Reuses tools/analyze.py's own load()/paired_effect() -- no custom scoring logic.
Both agents' rows come from ONE evaluate.py CSV (same literal rounds), so this
pairs on the exact game instance, not just on a replayed seed: the cleanest of
the three comparisons, with no stdlib-RNG cross-process noise floor at all.

Run: uv run python scratchpad/paired_same_board.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from analyze import load, paired_effect, METRICS  # noqa: E402

csv_path = REPO / "results/eval/task4_tournament/AgentA_AgentB_2rb__task4_headtohead_seed20260731.csv"
rows = load(csv_path)
rows_a = [r for r in rows if r["agent"] == "Agent_A"]
rows_b = [r for r in rows if r["agent"] == "Agent_B"]
print(f"n rows_a={len(rows_a)} rows_b={len(rows_b)}")

metrics = ["score", "coins", "kills", "suicides", "killed_by", "survived", "think_ms"]
print(f"\n{'metric':<10} {'A mean':>10} {'B mean':>10} {'diff (B-A)':>12} {'95% CI':>24} {'t':>7} {'p_flip':>7} verdict")
for metric in metrics:
    r = paired_effect(rows_a, rows_b, metric, n_boot=10_000)
    column = METRICS[metric][0]
    a_mean = sum(x[column] for x in rows_a) / len(rows_a)
    b_mean = sum(x[column] for x in rows_b) / len(rows_b)
    fragile = " (fragile)" if r["fragile"] else ""
    print(f"{metric:<10} {a_mean:>10.4f} {b_mean:>10.4f} {r['diff']:>+12.4f} "
          f"[{r['low']:+.4f}, {r['high']:+.4f}]   {r['t']:>6.2f} {r['p_flip']:>7.4f} {r['verdict']}{fragile}")
