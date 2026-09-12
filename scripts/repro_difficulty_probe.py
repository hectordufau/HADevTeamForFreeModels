import sys, os
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')

from harness.validation import runner as r
import harness.validation.field_validation as fv

# Show difficulty progression across successive load_benchmark_tasks calls
def show_diffs(label):
    vals = {}
    for split, attr in (("train","TRAIN_TASKS"),("validation","VALIDATION_TASKS"),("test","TEST_TASKS")):
        for t in getattr(r, attr):
            vals[t["id"]] = t.get("difficulty")
    from collections import Counter
    c = Counter(vals.values())
    print(f"{label}: difficulty histogram = {dict(c)}")

show_diffs("FRESH corpus before any load")
a = fv.load_benchmark_tasks()
show_diffs("after load call #1")
b = fv.load_benchmark_tasks()
show_diffs("after load call #2")
c = fv.load_benchmark_tasks()
show_diffs("after load call #3")

# Check individual: TRAIN-001 originally medium(0.50)
print("\nTRAIN-001 difficulty across calls: call1=%s call2=%s call3=%s" % (
    a["train"][0]["difficulty"], b["train"][0]["difficulty"], c["train"][0]["difficulty"]))
# TRAIN-004 originally high(0.72)
print("TRAIN-004 difficulty: call1=%s call2=%s call3=%s" % (
    a["train"][3]["difficulty"], b["train"][3]["difficulty"], c["train"][3]["difficulty"]))
