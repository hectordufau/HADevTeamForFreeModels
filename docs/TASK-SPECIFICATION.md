# DevTeamFree Task Specification

## 1. Metadata

**Title:**  
<Short descriptive task title>

**Type:**  
`FEATURE | BUGFIX | REFACTOR | ARCHITECTURE | SECURITY | PERFORMANCE | DOCUMENTATION | INVESTIGATION`

**Priority:**  
`LOW | MEDIUM | HIGH | CRITICAL`

**Target Project:**  
<Project/repository name>

**Target Branch:**  
<Branch if applicable>

---

## 2. Objective

Describe what must be accomplished and why.

Focus on the desired outcome rather than prescribing the implementation.

Example:

> Add authentication to the REST API so that protected endpoints can only be accessed by authenticated users.

---

## 3. Context

Provide the relevant business and technical context.

Include information such as:

- why the change is needed;
- existing behavior;
- affected subsystem;
- relevant architectural background;
- known dependencies;
- previous decisions that must be respected.

Do not duplicate information already available in authoritative project documentation unless necessary for understanding the task.

---

## 4. Functional Requirements

Define the required behavior.

Use stable identifiers.

### REQ-001 — <Requirement name>

**Description:**  
<Required behavior>

**Acceptance Criteria:**

- <criterion>
- <criterion>
- <criterion>

### REQ-002 — <Requirement name>

**Description:**  
<Required behavior>

**Acceptance Criteria:**

- <criterion>
- <criterion>

Add additional requirements as needed.

---

## 5. Non-Functional Requirements

Only include NFRs relevant to this task.

### NFR-001 — <NFR name>

**Category:**  
`SECURITY | PERFORMANCE | RELIABILITY | MAINTAINABILITY | OBSERVABILITY | COMPATIBILITY | SCALABILITY | OTHER`

**Requirement:**  
<Measurable or objectively verifiable requirement>

**Verification:**  
<How compliance can be demonstrated>

Example:

### NFR-001 — API response time

**Category:** PERFORMANCE

**Requirement:**  
The new endpoint must respond within 500 ms under the project's standard test workload.

**Verification:**  
Automated performance test.

---

## 6. Scope

### In Scope

- <item>
- <item>
- <item>

### Out of Scope

- <item>
- <item>

The agent must not implement out-of-scope functionality merely because it appears useful.

---

## 7. Constraints

List mandatory constraints.

Examples:

- preserve backward compatibility;
- do not change public API contracts;
- use existing project dependencies where possible;
- do not introduce paid services;
- do not weaken existing security controls;
- preserve database compatibility;
- follow existing architecture and coding standards.

---

## 8. Relevant References

List relevant project artifacts if known.

Examples:

- `README.md`
- `docs/architecture.md`
- `docs/ADR-003.md`
- `src/auth/`
- existing API specification;
- related issue/task;
- relevant test suite.

These are references, not automatic authorization to modify them.

---

## 9. Expected Behavior

Describe important end-to-end scenarios.

### Scenario 1 — <name>

**Given:**  
<context>

**When:**  
<action>

**Then:**  
<expected result>

### Scenario 2 — <name>

**Given:**  
<context>

**When:**  
<action>

**Then:**  
<expected result>

---

## 10. Error and Edge Cases

Describe known cases that require explicit handling.

- <edge case>
- <failure condition>
- <invalid input>
- <concurrency condition>
- <security condition>

The team should identify additional relevant edge cases during analysis.

---

## 11. Verification Requirements

The implementation must be verified using the mechanisms appropriate to the project and task.

Required where applicable:

- unit tests;
- integration tests;
- regression tests;
- build validation;
- lint/static analysis;
- security validation;
- acceptance criteria verification.

Existing tests must not be deleted, disabled, weakened, renamed to avoid discovery, or bypassed merely to make the implementation pass.

---

## 12. Definition of Done

The task is complete only when:

- all applicable functional requirements are satisfied;
- all applicable NFRs are satisfied;
- acceptance criteria pass;
- relevant existing tests continue to pass;
- new behavior has appropriate tests;
- no known security or governance constraint is violated;
- implementation remains within scope;
- required evidence is available;
- relevant documentation is updated;
- no unresolved blocking issue remains.

---

## 13. Deliverables

Expected outputs:

- implementation;
- tests;
- required configuration changes;
- documentation updates;
- verification evidence;
- summary of changes.

Add or remove deliverables according to the task.

---

## 14. Human Approval Required

Explicitly identify operations that require human approval before execution.

Examples:

- destructive database migrations;
- production deployment;
- removal of public APIs;
- security-policy changes;
- dependency changes with significant architectural impact;
- destructive operations;
- merge to protected branches;
- release/tag creation.

If uncertain whether an operation requires approval, stop before the irreversible operation and request human authorization.

---

## 15. Additional Notes

<Optional information, assumptions, examples, links, sample payloads, diagrams, or other useful context.>
