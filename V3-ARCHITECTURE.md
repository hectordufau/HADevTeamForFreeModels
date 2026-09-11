# V3.0 Architecture

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER / CLI                                 │
└──────────────────────┬───────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────────┐
│                      Orchestrator                                 │
│  (orchestrator.py — V2)                                          │
└───┬─────────┬──────────┬──────────┬──────────┬──────────────────┘
    │         │          │          │          │
    ▼         ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────────┐
│Task    │ │Planner │ │Workflow│ │Branch  │ │Policy Enforcer   │
│Intake  │ │        │ │Engine  │ │Manager │ │(V2 + V3)         │
│(V3 H)  │ │(V3 I)  │ │(V2)    │ │(V2.2)  │ └──────────────────┘
└────────┘ └────────┘ └────────┘ └────────┘
    │         │
    ▼         ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Execution Graph (V2.2)                         │
└───────┬───────────────────────────────────────┬──────────────────┘
        │                                       │
        ▼                                       ▼
┌────────────────┐                    ┌──────────────────┐
│ Agent Router   │                    │ Evidence Store   │
│ (V2.1 + V3 E)  │                    │ (V3 C)           │
│   ┌──────────┐ │                    └──────────────────┘
│   │Model     │ │                           │
│   │Intell.   │ │                    ┌──────▼───────┐
│   │(V3 F)    │ │                    │ Performance  │
│   └──────────┘ │                    │ Registry V3  │
└────────────────┘                    │ (Phase E)    │
                                      └──────────────┘
┌──────────────────────────────────────────────────────────────────┐
│                        Memory & Learning                         │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │Experience    │  │Failure         │  │Optimization        │   │
│  │Store (V3 D)  │  │Intell. (V3 J)  │  │Engine (V3 M)      │   │
│  └──────────────┘  └────────────────┘  └────────────────────┘   │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────────┐   │
│  │Context V3    │  │Observability   │  │Benchmark Engine    │   │
│  │(Phase K)     │  │(Phase N)       │  │(Phase B)           │   │
│  └──────────────┘  └────────────────┘  └────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                    Governance (V3 O)                              │
│  ImmutableGovernancePolicy · SecurityRegressionSuite             │
│  PromptInjectionBoundary · ToolAuthorizationHardener             │
└──────────────────────────────────────────────────────────────────┘
```

## Phases

| Phase | Module | Description |
|-------|--------|-------------|
| B | Benchmark Engine | Scenario registration, execution, metrics, reporting |
| C | Evidence V3 | Hash-chained evidence packages, canonical execution records |
| D | Memory V3 | Experience store, relevance scoring, confidence tracking |
| E | Performance V3 | Statistical performance aggregation, minimum sample policy, ranking |
| F | Model Intelligence | Adaptive model selection, exploration/exploitation, experiment tracking |
| G | Capability Intelligence | Synonym resolution, gap detection, recommendations, quality analysis |
| H | Task Intelligence | Task classification, validation, LLM-assisted analysis |
| I | Plan Intelligence | Plan scoring, alternative generation, selection, learning |
| J | Failure Intelligence | Root cause analysis, failure taxonomy, failure-to-learning pipeline, replanning |
| K | Context V3 | Context budget management, relevance scoring, priority-based inclusion |
| M | Optimization | Multi-target optimization, constraint enforcement, audit trail |
| N | Observability | Structured logging, decision tracing |
|| O | Policy V3 | Immutable governance, security regression, prompt injection, tool authorization |
|| P | Experience V3.1 | Structured experience extraction, quality scoring, evidence pipeline |
|| R | Retrieval V3.1 | Multi-factor retrieval, confidence calibration, historical usefulness, explainability |
|| S | Strategy V3.1 | Strategy generation from experiences, policy/capability/evidence validation |
|| T | Learning→Decision | Decision context, strategy-influenced selection, impact tracking |
|| U | Adaptive Explore | Adaptive rate (2-20%), purposeful candidate selection, result learning |
|| V | Evaluation V3.1 | Learning Gain, Generalization Gain, Failure Avoidance, Decision Influence |

## Key Design Decisions

1. **Free model invariant** — All modules enforce `cost=0`; paid models are blocked at every layer
2. **No hard-coded roles** — Capabilities, agents, and taxonomies are fully configurable
3. **Hash-chained integrity** — All evidence is chained with SHA-256 for tamper detection
4. **Immutable governance** — Policy rules cannot be modified at runtime
5. **Adaptive policy** — Model selection uses exploration/exploitation with confidence-aware ranking
