import sys, os, json
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')
from harness.validation.field_validation import (FieldValidationRunner,
    FieldExperimentConfig, MODES)
from scripts.phase8_field_validation import build_results

def full_run(reps, seed, run_id):
    cfg = FieldExperimentConfig(run_id=run_id, benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=seed, train_reps=reps, validation_reps=reps, test_reps=reps)
    r = FieldValidationRunner(cfg); r.run_all()
    res = build_results(r)
    return r, res

if __name__ == "__main__":
    repos = int(sys.argv[1])
    reseed = int(sys.argv[2])
    runid = sys.argv[3]
    r, res = full_run(repos, reseed, runid)
    out = {
        "run_id": runid,
        "reps": repos,
        "seed": reseed,
        "test": res["test"],
        "sample_sizes": res["sample_sizes"],
        "statistics_test": res["statistics_test"],
        "resolved_configuration": res["resolved_configuration"],
        "dataset_digest": res["dataset_digest"],
        "configuration_digest": res["configuration_digest"],
    }
    print(json.dumps(out, indent=2, sort_keys=True))
