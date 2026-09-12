import sys, os, json, hashlib, random
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')

from harness.validation import runner as r
import harness.validation.field_validation as fv

# --- Before/After mutation check ---
def snapshot(mod):
    out = {}
    for split, attr in (("train","TRAIN_TASKS"),("validation","VALIDATION_TASKS"),("test","TEST_TASKS")):
        for t in getattr(mod, attr):
            tid = t.get("id")
            out[(split,tid)] = {
                "difficulty": t.get("difficulty"),
                "category": t.get("category"),
            }
    return out

before = dict(snapshot(r))
print("Source corpus contains 'difficulty' key as string (e.g. 'low'/'medium'/'high') before load...")
has_cat = any("category" in v for v in before.values())
print("Any 'category' key present in source before load_benchmark_tasks():", has_cat)

# digest before
db_before = fv.dataset_digest()
print("dataset_digest BEFORE load_benchmark_tasks():", db_before)

# Call load_benchmark_tasks() once
tasks1 = fv.load_benchmark_tasks()

after = dict(snapshot(r))
mutation_count = 0
for k in before:
    if before[k]["difficulty"] != after[k]["difficulty"] or before[k]["category"] != after[k]["category"]:
        mutation_count += 1
print("\nMUTATION: number of task dicts in source corpus changed by load_benchmark_tasks():", mutation_count, "of", len(before))
print("Example before:", list(before.items())[0])
print("Example after :", list(after.items())[0])

# digest after one load
db_after1 = fv.dataset_digest()
print("\ndataset_digest AFTER load_benchmark_tasks():", db_after1)
print("digest stable?", db_before == db_after1)

# Call load_benchmark_tasks() twice more and inspect outputs differ
tasks2 = fv.load_benchmark_tasks()
tasks3 = fv.load_benchmark_tasks()
# Check if values are same across calls
t1 = tasks1["train"][0]
t3 = tasks3["train"][0]
print("\nTask[0] after call1:", t1["difficulty"], t1["category"])
print("Task[0] after call3:", t3["difficulty"], t3["category"])
print("Same object identity across calls for source? first task of call1 is source obj:", t1 is r.TRAIN_TASKS[0])

# Check values semantically identical across loads (difficulty always numeric now)
all_num = all(isinstance(t["difficulty"], (int,float)) for split in ("train","validation","test") for t in tasks1[split])
print("All difficulties numeric after load:", all_num)
