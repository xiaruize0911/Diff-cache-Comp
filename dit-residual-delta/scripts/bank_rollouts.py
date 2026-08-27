#!/usr/bin/env python
"""Print how many rollouts a collected bank contains, or 0 if it is incomplete.

collect_features.py writes manifest.json only after its loop finishes, so the
manifest's presence plus a matching rollout total is an exact "this bank is
done" marker. This lives in its own file because inlining it as a heredoc inside
a command substitution proved unreliable: in scripts/matched_ladder.sh the same
construct passed for the val split and failed for the train split of the same
bank, both manifests valid, and the failure silently wiped a finished 22 GB bank
twice. The asymmetry was never explained, so the construct is gone instead.
"""
import json, sys
from pathlib import Path

m = Path(sys.argv[1]) / "manifest.json"
if not m.exists():
    print(0)
else:
    try:
        print(sum(s["rollouts"] for s in json.loads(m.read_text())["shards"]))
    except Exception:
        print(0)
