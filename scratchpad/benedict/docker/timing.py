"""Per-step think-time harness. Mirrors tools/evaluate.py's instrumentation
(patch Agent.note_stat in memory) so the numbers are comparable, but depends on
nothing outside the framework -- tools/ is not in the submission."""
import os, sys, json
sys.path.insert(0, os.getcwd())
import numpy as np
from agents import Agent
import settings as s

TARGET = "benedict_task4"
times = []
_orig = Agent.note_stat

def _patched(self, name, value=1):
    if name == "time" and self.name.startswith(TARGET):
        times.append(float(value))
    return _orig(self, name, value)

Agent.note_stat = _patched

sys.argv = ["main.py", "play",
            "--agents", TARGET, "rule_based_agent", "rule_based_agent", "rule_based_agent",
            "--scenario", "classic", "--no-gui", "--n-rounds", sys.argv[1]]
import main
main.main()

t = np.array(times) * 1000.0          # ms
print(json.dumps({
    "steps":        int(t.size),
    "mean_ms":      round(float(t.mean()), 4),
    "median_ms":    round(float(np.median(t)), 4),
    "p99_ms":       round(float(np.percentile(t, 99)), 4),
    "p99_9_ms":     round(float(np.percentile(t, 99.9)), 4),
    "max_ms":       round(float(t.max()), 4),
    "limit_ms":     1000 * s.TIMEOUT,
    "over_limit":   int((t > 1000 * s.TIMEOUT).sum()),
    "headroom_x":   round(float(1000 * s.TIMEOUT / t.max()), 1),
}, indent=2))
