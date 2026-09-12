import sys, os, json
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')
from harness.validation.field_validation import FieldValidationRunner, FieldExperimentConfig, MODES

def run(reps, seed, run_id):
    cfg = FieldExperimentConfig(run_id=run_id, benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=seed, train_reps=reps, validation_reps=reps, test_reps=reps)
    r = FieldValidationRunner(cfg); r.run_all()
    # full observation signature for exact comparison
    sig = []
    for o in sorted(r.observations, key=lambda x:(x.mode,x.split,x.task_id,x.repetition)):
        sig.append((o.mode,o.split,o.task_id,o.repetition,round(o.score,6),o.status,o.derived_seed))
    return sig

# TWO runs in the SAME process (I5: repeated runs same process)
a1 = run(5, 314159, "V3.2-FV8-20260911")
a2 = run(5, 314159, "V3.2-FV8-20260911")
print("Same-process identical (I5):", a1 == a2)

# Then a completely fresh process comparison is done separately.
print("num obs:", len(a1))
print("unique:", len(set(a1)))
