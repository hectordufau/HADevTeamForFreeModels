# Hermes Agent DevTeam For Free Models

> AI engineering harness for Hermes Agent: capability-driven execution, free models, verification, and independent review.

## What it does

The harness receives a software-engineering demand, identifies the required capabilities, selects compatible agent profiles, plans a dependency-aware workflow, executes it, and verifies the result.

```text
Task → Capabilities → Agents → Workflow → Execution → Verification → Review
```

The workflow is driven by declared capabilities, not by a fixed manager/architect/coder/tester sequence. Roles remain metadata; capabilities and policy determine routing.

### Execution guarantees

- Formal task contracts with objectives, acceptance criteria, and allowed changes
- Capability matching through `CapabilityRegistry` and `CapabilityMatcher`
- Declarative capability dependencies in `config/capability_taxonomy.yaml`
- DAG validation and dependency-ordered execution
- Verification and acceptance-criteria evaluation
- Independent reviewer gate
- Free-model policy enforcement
- Failure analysis, replanning, and bounded retries
- Evidence and learning artifacts with run/mode/task attribution

## Quick start

```bash
git clone https://github.com/hectordufau/HADevTeamForFreeModels.git
cd HADevTeamForFreeModels
python3 setup.py
```

Configure Hermes with the Nous Portal free-model provider, then run from Hermes chat:

```text
/devteamfree <your demand>
```

## Requirements

- Hermes Agent
- Nous Portal account using the free-model tier
- `hermes setup --portal`
- Python 3 for the setup and test tooling

## Current release: V3.3.0

V3.3.0 adds **Engineering Knowledge & Decision Intelligence** to the engineering lifecycle. It is released and immutable at tag `v3.3.0`.

Engineering Knowledge is structured separately from Learning Knowledge and can inform decisions without becoming an authority or bypassing governance:

```text
Knowledge → Retrieval → Planning → Policy → Execution → Verification → Evidence → Review → Learning
```

### Engineering Knowledge coverage

- Requirements and non-functional requirements: PRD, REQ, NFR
- Design and architecture decisions: DR, ADR, TDR
- Risk and security records: RSK, SEC
- Root-cause and failure intelligence: RCA
- Knowledge storage, graph, retrieval, context assembly, and traceability
- Reviewer and evidence integration
- Knowledge-to-learning bridge with explicit authority boundaries
- Governance and policy enforcement through the existing policy architecture

### V3.3 boundaries

- Authoritative engineering knowledge is distinct from learned experience and exploration
- Learning is advisory and cannot bypass security, governance, policy, or authority rules
- Knowledge presence, retrieval, influence, compliance, and beneficial outcome are separate measurements
- Free-model, evidence, review, and historical-artifact protections remain enforced

V3.3.0 passed **1960/1960 tests** at release. Phase 13's historical controlled validation result was **NEUTRAL**; release correctness and empirical effectiveness are separate claims.

## Phase 16 field validation

Phase 16 is a post-release experiment. It does not modify or re-release V3.3.0.

The accepted run used a frozen synthetic benchmark with:

- 20 independent task clusters
- 5 tasks in each knowledge-sensitivity stratum: HIGH, MEDIUM, LOW, NONE
- 3 modes: `CONTROL`, `KNOWLEDGE_ONLY`, `KNOWLEDGE_PLUS_LEARNING`
- 10 repeated observations per task and mode
- 600 total executions; repetitions were not treated as independent tasks
- Task-level paired analysis with 10,000 bootstrap and 10,000 permutation iterations

The first accepted run was invalidated because of two harness defects. Its raw evidence remains preserved. Corrected run `V3.3-P16-FV-20260918-R2` passed the purity and reproducibility checks and produced a **NEUTRAL** result for the instrumented synthetic score. Latency, cost, safety, and governance outcomes were not instrumented and are reported as unavailable rather than zero. The result is scoped to this benchmark and configuration; it is not a production-effectiveness claim.

Phase 16 does not authorize a merge to `master`, a new release, or a product version bump.

## Documentation

### Architecture and implementation

| Document | Purpose |
|---|---|
| [`SPEC-V2.md`](./SPEC-V2.md) | V2 architectural specification |
| [`SPEC-V2.2.md`](./SPEC-V2.2.md) | Capability-driven workflow specification |
| [`IMPLEMENTATION-PLAN-V2.md`](./IMPLEMENTATION-PLAN-V2.md) | Ordered implementation plan |
| [`docs/V3.3-SPEC.md`](./docs/V3.3-SPEC.md) | V3.3 authoritative specification |
| [`docs/V3.3-MASTER-IMPLEMENTATION-PLAN.md`](./docs/V3.3-MASTER-IMPLEMENTATION-PLAN.md) | V3.3 implementation plan |

### Release and validation evidence

| Document | Purpose |
|---|---|
| [`docs/V3.3-RELEASE-NOTES.md`](./docs/V3.3-RELEASE-NOTES.md) | User-facing V3.3 release notes |
| [`docs/V3.3-RELEASE-REPORT.md`](./docs/V3.3-RELEASE-REPORT.md) | V3.3.0 release verification and boundaries |
| [`docs/V3.3-RELEASE-MANIFEST.json`](./docs/V3.3-RELEASE-MANIFEST.json) | Release identity and integrity manifest |
| [`docs/V3.3-PHASE16-SPEC.md`](./docs/V3.3-PHASE16-SPEC.md) | Phase 16 experiment specification |
| [`docs/V3.3-PHASE16-EXPERIMENT-MANIFEST.json`](./docs/V3.3-PHASE16-EXPERIMENT-MANIFEST.json) | Frozen/corrected experiment identity and digests |
| [`docs/V3.3-PHASE16-FIELD-VALIDATION-REPORT.md`](./docs/V3.3-PHASE16-FIELD-VALIDATION-REPORT.md) | Accepted field-validation report |

Historical V3.1, V3.2, and per-phase implementation reports remain in `docs/` for audit and reproducibility. The README keeps only the user-facing summary.

## Agent profiles

Profiles declare capabilities in `PROFILE.yaml`. The harness discovers them automatically.

| Profile | Typical role | Example capabilities |
|---|---|---|
| `devfree-manager` | Triage/orchestration | `orchestration`, `task_classification` |
| `devfree-architect` | Architecture/security | `architecture`, `system_design`, `security` |
| `devfree-coder` | Implementation | `coding`, `refactoring`, `testing` |
| `devfree-tester` | Tests/debugging | `testing`, `debugging`, `verification` |
| `devfree-reviewer` | Independent review | `code_review`, `reasoning`, `security` |
| `devfree-worker` | Focused work | `coding`, `debugging` |

Add a specialist by creating a profile with its declared capabilities. Routing uses those declarations rather than a hard-coded fallback.

## Tests

Run the full suite with:

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
```

The V3.3.0 release baseline is `1960/1960 PASS`. Phase 16 adds experimental infrastructure and validation tests without changing the released tag.

## Version history

```text
v2.1.0  Adaptive Harness + Integrity
v2.2.0  Dynamic Capability-Driven Workflow
v2.2.1  Capability Integrity Patch
v3.0.0  Learning + Optimization Engineering Harness
v3.1.0  Learning Optimization & Decision Intelligence
v3.2.0  Learning Integrity, Causal Attribution & Experimental Reproducibility
v3.3.0  Engineering Knowledge & Decision Intelligence
```

## License

MIT
