import sys, os, json
sys.path.insert(0, '/home/hector/workspace/devteamfree-publish')
os.chdir('/home/hector/workspace/devteamfree-publish')
from harness.validation.field_validation import (
    FieldValidationRunner, FieldExperimentConfig, configuration_digest,
    dataset_digest, protected_release_results_path, repo_root)
from scripts.phase8_field_validation import build_results

# New final run: fixed code, fresh process, fresh validation_run_id, reps 5, seed 314159.
FINAL_RUN_ID = "V3.2-FV8-20260912-reprod"
PRIOR_RUN_ID = "V3.2-FV8-20260911"

def full_run(reps, seed, run_id):
    cfg = FieldExperimentConfig(run_id=run_id, benchmark_id="V3.2-FIELD-VALIDATION",
        reproducibility_seed=seed, train_reps=reps, validation_reps=reps, test_reps=reps)
    r = FieldValidationRunner(cfg); r.run_all()
    return r, build_results(r)

r, results = full_run(5, 314159, FINAL_RUN_ID)

# Add the explicit supersedes marker (release-evidence replacement protocol).
results["supersedes"] = {
    "prior_validation_run_id": PRIOR_RUN_ID,
    "reason": ("Reproducibility defect corrected: load_benchmark_tasks() no longer "
               "mutates the shared benchmark corpus in place (idempotency fix) and "
               "repetition count is now canonicalized to a single source. Prior "
               "accepted run validated as reproducible with its exact recorded "
               "configuration; this run re-establishes final release evidence on the "
               "corrected, reproducible codebase."),
}
# Persist the new durable release evidence.
out_dir = os.path.join(repo_root(), "artifacts", "v3.2-release", "field_validation")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "results_REPROD_20260912.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
print("WROTE:", path)
print("FINAL_RUN_ID:", FINAL_RUN_ID)
print("test:", json.dumps(results["test"], sort_keys=True))
print("sample_sizes_test:", json.dumps(results["sample_sizes"]["test"], sort_keys=True))
print("dataset_digest:", results["dataset_digest"])
print("configuration_digest:", results["configuration_digest"])
print("code_revision:", results["code_revision"])
print("resolved_configuration:", json.dumps(results["resolved_configuration"], indent=2))
print("supersedes:", json.dumps(results["supersedes"], indent=2))
