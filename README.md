# HADevTeamForFreeModels

> An AI Engineering Harness for Hermes Agent — free models, verifiable execution, autonomous software engineering.

## What is this?

A set of 6 Hermes profiles that work in a pipeline to receive software engineering demands, triage, implement, test, and review code — using **only free models from the Nous Portal**.

**V2.0** transforms this into an **AI Engineering Harness** with formal task contracts, state machines, context engineering, verification, evidence, evaluation, and controlled iteration.

## Quick Start

```bash
git clone https://github.com/hectordufau/HADevTeamForFreeModels.git
cd HADevTeamForFreeModels
python3 setup.py
```

Then in Hermes chat:

```
/devteamfree <your demand>
```

## V2.0 Documentation

| Document | Purpose |
|---|---|
| [`SPEC-V2.md`](./SPEC-V2.md) | Full architectural specification |
| [`IMPLEMENTATION-PLAN-V2.md`](./IMPLEMENTATION-PLAN-V2.md) | Ordered implementation tasks for autonomous execution |
| [`setup.py`](./setup.py) | Generates profiles from current Nous Portal FREE models |

## Profiles

| Profile | Role | Default FREE Model |
|---|---|---|
| `devfree-manager` | Triage/orchestration | Meituan LongCat 2.0 |
| `devfree-architect` | Architecture/security | Poolside Laguna S 2.1 |
| `devfree-coder` | Implementation | Meituan LongCat 2.0 |
| `devfree-tester` | Tests/debug | StepFun Step 3.7 Flash |
| `devfree-reviewer` | Code review | Poolside Laguna S 2.1 |
| `devfree-worker` | Simple tasks | Upstage Solar Pro 4 |

## How to implement V2.0

An autonomous agent can implement the full V2.0 specification by reading `SPEC-V2.md` + `IMPLEMENTATION-PLAN-V2.md` and executing the 15 tasks sequentially. Each task has defined acceptance criteria, verification steps, and scope boundaries.

See [`IMPLEMENTATION-PLAN-V2.md`](./IMPLEMENTATION-PLAN-V2.md) for the autonomous agent directive.

## Requirements

- Hermes Agent
- Nous Portal account (free)
- `hermes setup --portal` configured

## License

MIT
