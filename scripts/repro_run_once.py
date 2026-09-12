import sys, os, json
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')

from harness.validation.field_validation import FieldValidationRunner, FieldExperimentConfig, MODES

def run(reps, seed, run_id):
    cfg = FieldExperimentConfig(
        run_id=run_id, benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=seed, train_reps=reps, validation_reps=reps, test_reps=reps,
    )
    r = FieldValidationRunner(cfg)
    r.run_all()
    # TEST split per-mode aggregates
    out = {}
    for m in MODES:
        o = r.observations_for_split(m, "test")
        sr = sum(1 for x in o if x.status == "PASSED") / len(o)
        sc = sum(x.score for x in o) / len(o)
        out[m] = {"n": len(o), "success": round(sr,4), "score": round(sc,4)}
    return out

res = run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
print(json.dumps(res, indent=2))
