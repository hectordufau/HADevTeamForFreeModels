# harness/orchestrator/__init__.py — Orchestrator and End-to-End Harness (V2.2)
"""
The Orchestrator is the central coordination component.
Controls the complete task lifecycle from intake to delivery.
V2.2: uses capability-driven planner pipeline when enabled.
V2.1 fallback via config flag `use_v21_workflow`.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import os
import yaml
import asyncio


class OrchestratorError(Exception):
    """Raised on orchestration failures."""


class Orchestrator:
    """Coordinates the complete task lifecycle with all V2.1/V2.2 integrations."""

    def __init__(
        self,
        config: dict,
        state_manager: Any,
        task_manager: Any,
        agent_loader: Any,
        model_router: Any,
        context_manager: Any,
        agent_executor: Any,
        verification_engine: Any,
        evidence_collector: Any,
        evaluation_engine: Any,
        iteration_engine: Any,
        # V2.1 additions
        workflow_engine: Optional[Any] = None,
        policy_engine: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        performance_registry: Optional[Any] = None,
        adaptive_router: Optional[Any] = None,
        # V2.2 additions
        capability_registry: Optional[Any] = None,
        capability_taxonomy: Optional[Any] = None,
        task_analyzer: Optional[Any] = None,
        workflow_planner: Optional[Any] = None,
        workflow_validator: Optional[Any] = None,
        execution_graph: Optional[Any] = None,
        failure_analyzer: Optional[Any] = None,
        replanner: Optional[Any] = None,
        branch_resolver: Optional[Any] = None,
        workspace_manager: Optional[Any] = None,
    ):
        self.config = config
        self.state = state_manager
        self.tasks = task_manager
        self.agents = agent_loader
        self.router = model_router
        self.context = context_manager
        self.executor = agent_executor
        self.verifier = verification_engine
        self.evidence = evidence_collector
        self.evaluator = evaluation_engine
        self.iteration = iteration_engine
        # V2.1
        self.workflow = workflow_engine
        self.policy = policy_engine
        self.memory = memory_manager
        self.registry = performance_registry
        self.adaptive_router = adaptive_router
        # V2.2
        self.cap_registry = capability_registry
        self.cap_taxonomy = capability_taxonomy
        self.task_analyzer = task_analyzer
        self.workflow_planner = workflow_planner
        self.workflow_validator = workflow_validator
        self.execution_graph = execution_graph
        self.failure_analyzer = failure_analyzer
        self.replanner = replanner
        self.branch_resolver = branch_resolver
        self.workspace_manager = workspace_manager
        self._artifacts_dir = "artifacts/executions"

    def run(self, task_id: str, workflow_name: str = "default") -> Dict[str, Any]:
        """
        Execute a complete task lifecycle.

        Uses V2.2 capability-driven pipeline if all components are configured,
        otherwise falls back to V2.1 WorkflowEngine or hard-coded steps.

        Returns the final execution record with complete artifacts.
        """
        # Load task contract
        task = self.tasks.load(os.path.join("tasks", "active", f"{task_id}.yaml"))

        # Register state
        state = self.state.register(task_id)
        state.transition("ANALYZING")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        # Load context from memory if available
        if self.memory:
            memory_context = self.memory.search(task.spec.objective, category="lessons")
        else:
            memory_context = []

        # Determine which pipeline to use
        use_v22 = self._should_use_v22()
        use_v21_workflow = self._should_use_v21_workflow()

        if use_v22:
            return self._run_v22_pipeline(task_id, task, state, memory_context)
        elif use_v21_workflow:
            return self._run_v21_workflow_pipeline(task_id, task, state, memory_context)
        else:
            return self._run_v21_hardcoded_pipeline(task_id, task, state, memory_context)

    def _should_use_v22(self) -> bool:
        """Check if V2.2 pipeline should be used (all components present)."""
        return all([
            self.cap_registry,
            self.task_analyzer,
            self.workflow_planner,
            self.workflow_validator,
            self.execution_graph is not None or True,  # optional in pipeline
            not self.config.get("use_v21_workflow", False),
        ])

    def _should_use_v21_workflow(self) -> bool:
        """Check if V2.1 WorkflowEngine should be used."""
        wf_enabled = self.config.get("use_v21_workflow", False)
        if isinstance(wf_enabled, str):
            wf_enabled = wf_enabled.lower() == "true"
        return bool(wf_enabled and self.workflow)

    def _run_v22_pipeline(self, task_id: str, task: Any, state: Any,
                          memory_context: List[Any]) -> Dict[str, Any]:
        """Execute the V2.2 capability-driven pipeline."""
        # State is already ANALYZING (set in run())
        # transition to PLANNING is the next step
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        # Step 1: Analyze task -> capability requirements
        try:
            requirements = self.task_analyzer.analyze(task)
        except Exception as e:
            state.transition("FAILED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            return {"task_id": task_id, "status": "FAILED", "error": f"Task analysis failed: {e}"}

        self._save_artifact(task_id, "capability_requirements.yaml", {
            "required": requirements.required,
            "optional": requirements.optional,
            "constraints": requirements.constraints,
            "agent_preferences": requirements.agent_preferences,
        })

        # Step 2: Plan workflow from requirements
        state.transition("PLANNED")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        try:
            workflow = self.workflow_planner.plan(requirements, task=task)
        except Exception as e:
            state.transition("FAILED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            return {"task_id": task_id, "status": "FAILED", "error": f"Workflow planning failed: {e}"}

        self._save_artifact(task_id, "workflow_plan.yaml", workflow.to_dict())

        # Step 3: Validate workflow (stays in PLANNED state)
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        try:
            validation = self.workflow_validator.validate(workflow, task=task)
        except Exception as e:
            state.transition("FAILED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            return {"task_id": task_id, "status": "FAILED", "error": f"Workflow validation failed: {e}"}

        self._save_artifact(task_id, "workflow_validation.yaml", {
            "valid": validation.valid,
            "issues": [{"code": i.code, "message": i.message, "node_id": i.node_id}
                      for i in validation.issues],
        })

        if not validation.valid:
            state.transition("BLOCKED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            return {"task_id": task_id, "status": "BLOCKED",
                    "error": f"Workflow validation failed: {[i.code for i in validation.issues]}"}

        # Step 4: Build capability graph from workflow
        cap_graph = self._build_capability_graph(workflow)
        self._save_artifact(task_id, "capability_graph.yaml", cap_graph.to_dict())

        # Step 5: Execute workflow via ExecutionGraph
        state.transition("IMPLEMENTING")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
        self._current_task = task  # Make task available to ExecutionGraph

        max_attempts = self.config.get("execution", {}).get("max_iterations", 3)
        attempt = 0
        final_result = None

        while attempt < max_attempts:
            attempt += 1

            try:
                # Instantiate ExecutionGraph with the workflow and orchestrator
                graph = self.execution_graph(workflow, self) if isinstance(self.execution_graph, type) else self.execution_graph
                exec_result = asyncio.run(graph.execute())
            except Exception as e:
                state.transition("FAILED")
                self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
                return {"task_id": task_id, "status": "FAILED", "error": f"Execution failed: {e}"}

            self._save_artifact(task_id, "execution_graph.yaml", exec_result)

            # Check if execution succeeded
            if exec_result["status"] == "completed":
                final_result = exec_result
                break

            # Check if we have failure analysis and replanning
            if not self.failure_analyzer or not self.replanner:
                final_result = exec_result
                break

            # Analyze failures
            # Use a try-except to handle state machine transitions gracefully
            try:
                state.transition("VERIFYING")
            except Exception:
                pass  # State machine doesn't support this transition
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

            failure_reports = []
            for fnid in exec_result["failed_nodes"]:
                fn = workflow.get_node(fnid)
                if fn:
                    node_result_data = exec_result.get("node_results", {}).get(fnid, {})
                    report = self.failure_analyzer.analyze(
                        fn,
                        {"status": "failed", "error": node_result_data.get("error", "unknown")},
                    )
                    failure_reports.append(report)

            if failure_reports:
                self._save_artifact(task_id, "failure_analysis.yaml", {
                    "reports": [
                        {
                            "node_id": r.node_id,
                            "capability": r.capability,
                            "category": r.category,
                            "message": r.message,
                            "missing_capabilities": r.missing_capabilities,
                            "recommendations": r.recommendations,
                        }
                        for r in failure_reports
                    ]
                })

            # Replan
            try:
                state.transition("ITERATING")
            except Exception:
                pass
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

            try:
                completed_nodes = exec_result.get("completed_nodes", [])
                failed_node = failure_reports[0].node_id if failure_reports else ""
                report = failure_reports[0] if failure_reports else None

                if report:
                    new_workflow = self.replanner.replan(
                        task, workflow, completed_nodes, failed_node, report
                    )

                    # Save replan report
                    r = self.replanner.generate_replan_report(
                        workflow, new_workflow,
                        f"Failure recovery from {report.category}: {report.message}"
                    )
                    self._save_artifact(task_id, "workflow_replan.yaml", r)

                    # Use the new workflow going forward
                    workflow = new_workflow
                else:
                    final_result = exec_result
                    break

            except Exception as e:
                final_result = exec_result
                break

        if final_result is None:
            final_result = exec_result

        # --- Final state ---
        final_status = "COMPLETED" if final_result["status"] == "completed" else "FAILED"
        try:
            state.transition(final_status)
        except Exception:
            # State may not be a valid transition, just set the state directly
            state.current_state = final_status

        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        # Build report
        report = self._build_report(task_id, state, None, None)
        report["execution"] = final_result
        self._save_artifact(task_id, "final_report.yaml", report)

        return report

    def _build_capability_graph(self, workflow: Any):
        """Build a CapabilityGraph from an ExecutionWorkflow."""
        from harness.capabilities.graph import CapabilityGraph, CapabilityNode

        cg = CapabilityGraph()
        for node in workflow.nodes:
            cg.add_node(CapabilityNode(
                id=node.id,
                capability=node.capability,
                dependencies=list(node.depends_on),
                status="pending",
            ))
        return cg

    def _run_v21_workflow_pipeline(self, task_id: str, task: Any, state: Any,
                                   memory_context: List[Any],
                                   workflow_name: str = "default") -> Dict[str, Any]:
        """Execute V2.1 WorkflowEngine pipeline (unchanged)."""
        wf_path = os.path.join("config", "workflow_definitions", f"{workflow_name}.yaml")
        if os.path.exists(wf_path):
            workflow = self.workflow.load_definition(wf_path)
            self._save_artifact(task_id, "workflow.yaml",
                                {"name": workflow.name, "steps": [s.role for s in workflow.steps]})
            wf_result = self.workflow.execute(workflow, task)

            # Record performance data
            if self.registry:
                for step_result in wf_result["steps"]:
                    self.registry.record(
                        self.registry._record_type(
                            task_id=task_id,
                            model_id=step_result.get("model", "unknown"),
                            role=step_result["step"],
                            success=step_result["status"] == "completed",
                            score=1.0 if step_result["status"] == "completed" else 0.0,
                            iterations=1,
                            latency_ms=0.0,
                            capabilities_used=[],
                        )
                    )

            final_state = "COMPLETED" if wf_result["status"] == "completed" else "FAILED"
            state.transition(final_state)
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

            report = self._build_report(task_id, state, None, None)
            self._save_artifact(task_id, "final_report.yaml", report)
            return report

        # Fall through to hard-coded if no workflow definition found
        return self._run_v21_hardcoded_pipeline(task_id, task, state, memory_context)

    def _run_v21_hardcoded_pipeline(self, task_id: str, task: Any, state: Any,
                                    memory_context: List[Any]) -> Dict[str, Any]:
        """Execute the original V2.1 hard-coded pipeline (preserved for regression)."""
        # Policy check before manager execution
        if self.policy:
            policy_result = self.policy.evaluate_all(
                agent_level=2, task_level=task.spec.autonomy.get("maximum", 3), tool_level=2,
                current_iteration=0, has_verification=False,
            )
            if not policy_result.allowed:
                state.transition("BLOCKED")
                self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
                return {"task_id": task_id, "status": "BLOCKED",
                        "error": f"Policy violation: {policy_result.violations}"}

        # --- Manager phase ---
        self._run_agent_phase(task_id, task, state, "manager",
                              self._infer_capabilities(task), context_pack_extra=memory_context)

        # --- Architect phase ---
        self._run_agent_phase(task_id, task, state, "architect",
                              ["architecture", "reasoning", "code_analysis"])

        # --- Implementation phase ---
        state.transition("IMPLEMENTING")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        coder_agent = self.agents.load("coder")
        coder_selection = self._select_model_with_fallback(task_id, ["coding", "debugging"], "coder")
        coder_context = self.context.build_context(
            task_contract=task, agent_role="coder",
            agent_soul=coder_agent.soul_content,
            relevant_files=task.spec.allowed_changes,
        )

        coder_result = self.executor.execute(
            coder_agent, coder_selection.model_id, task, coder_context, tool_name="filesystem_write"
        )
        if coder_result.status == "failed":
            state.transition("FAILED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            return {"task_id": task_id, "status": "FAILED", "error": coder_result.error}

        # --- Verification ---
        state.transition("VERIFYING")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
        verif_result = self.verifier.verify(task, workspace=".", checks=task.spec.verification.required)
        self._save_artifact(task_id, "verification.yaml",
                            {"status": verif_result.status,
                             "checks": {k: {"status": v.status, "passed": v.passed, "failed": v.failed}
                                        for k, v in verif_result.checks.items()}})

        # --- Evidence ---
        evidence_record = self.evidence.collect(coder_result, task_id)
        self.evidence.persist(evidence_record)
        self._save_artifact(task_id, "evidence.yaml", evidence_record.to_dict())

        # --- Evaluation ---
        state.transition("EVALUATING")
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
        eval_result = self.evaluator.evaluate(task, verif_result, evidence_record)
        self._save_artifact(task_id, "evaluation.yaml",
                            {"score": eval_result.score, "decision": eval_result.decision,
                             "acceptance": [{"criterion": a.criterion, "status": a.status}
                                           for a in eval_result.acceptance]})

        # --- Iteration loop ---
        max_attempts = self.config.get("execution", {}).get("max_iterations", 3)
        attempt = 1
        while not eval_result.is_passed() and attempt < max_attempts:
            has_security_failure = any(
                "security" in a.criterion.lower() and a.status == "failed"
                for a in eval_result.acceptance
            )
            if has_security_failure:
                state.transition("NEEDS_HUMAN")
                self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
                break

            attempt += 1
            feedback = self.iteration.build_feedback(task, eval_result, verif_result)

            if not self.iteration.should_iterate(attempt, feedback):
                state.transition("NEEDS_HUMAN" if feedback.failures else "FAILED")
                self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
                break

            state.transition("ITERATING")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            self._save_artifact(task_id, f"feedback.yaml", {
                "attempt": attempt,
                "failures": [{"check": f.check, "reason": f.reason, "class": f.failure_class}
                            for f in feedback.failures],
                "recommendations": feedback.recommendations,
                "target_agent": feedback.target_agent,
            })

            coder_context.memory_notes.append(f"Attempt {attempt}: {feedback.recommendations}")
            coder_result = self.executor.execute(
                coder_agent, coder_selection.model_id, task, coder_context, tool_name="filesystem_write"
            )
            verif_result = self.verifier.verify(task, workspace=".", checks=task.spec.verification.required)
            evidence_record = self.evidence.collect(coder_result, task_id)
            eval_result = self.evaluator.evaluate(task, verif_result, evidence_record)

        # --- Review phase ---
        if state.current_state not in {"FAILED", "CANCELLED", "NEEDS_HUMAN", "COMPLETED", "BLOCKED"}:
            state.transition("REVIEWING")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

            reviewer_agent = self.agents.load("reviewer")
            reviewer_selection = self._select_model_with_fallback(
                task_id, ["code_review", "reasoning", "security"], "reviewer"
            )

            reviewer_context = self.context.build_context(
                task_contract=task, agent_role="reviewer",
                agent_soul=reviewer_agent.soul_content,
                relevant_files=task.spec.allowed_changes,
            )
            reviewer_context.memory_notes = [
                f"Task: {task.spec.objective}",
                f"Verification: {verif_result.status}",
                f"Evaluation score: {eval_result.score}",
                f"Evaluation decision: {eval_result.decision}",
            ]

            reviewer_result = self.executor.execute(
                reviewer_agent, reviewer_selection.model_id, task, reviewer_context,
                tool_name="filesystem_read"
            )

            self._save_artifact(task_id, "review.yaml", {
                "model": reviewer_selection.model_id,
                "status": reviewer_result.status,
                "claims": reviewer_result.claims,
            })

            if eval_result.is_passed() and reviewer_result.status == "completed":
                state.transition("COMPLETED")
            else:
                state.transition("FAILED")

            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        # --- Record performance ---
        if self.registry:
            self.registry.record(
                self.registry._record_type(
                    task_id=task_id,
                    model_id=coder_selection.model_id,
                    role="coder",
                    success=state.current_state == "COMPLETED",
                    score=eval_result.score if hasattr(eval_result, 'score') else 0.0,
                    iterations=attempt,
                    latency_ms=0.0,
                    capabilities_used=["coding", "debugging"],
                )
            )

        # --- Store lessons in memory ---
        if self.memory and state.current_state == "COMPLETED":
            self.memory.store(
                self.memory._entry_type(
                    key=f"lesson-{task_id}",
                    content=f"Task '{task.spec.objective}' completed with score {eval_result.score}",
                    category="lessons",
                    tags=["completed", task.spec.objective[:30]],
                )
            )

        # --- Final report ---
        report = self._build_report(task_id, state, eval_result, evidence_record)
        self._save_artifact(task_id, "final_report.yaml", report)

        return report

    def _run_agent_phase(self, task_id: str, task: Any, state: Any,
                         role: str, capabilities: List[str],
                         context_pack_extra: Optional[list] = None):
        """Execute a single agent phase with policy checks."""
        state.transition(role.upper() if role == "manager" else "PLANNED" if role == "architect" else role.upper())
        self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))

        agent = self.agents.load(role)
        selection = self._select_model_with_fallback(task_id, capabilities, role)

        ctx = self.context.build_context(
            task_contract=task, agent_role=role,
            agent_soul=agent.soul_content,
        )
        if context_pack_extra:
            ctx.memory_notes = [str(m.content) for m in context_pack_extra]

        result = self.executor.execute(agent, selection.model_id, task, ctx, tool_name="filesystem_read")
        if result.status == "failed":
            state.transition("FAILED")
            self._save_artifact(task_id, "state.yaml", self._state_to_dict(state))
            raise OrchestratorError(f"{role} phase failed: {result.error}")

    def _select_model_with_fallback(self, task_id: str, capabilities: List[str], role: str) -> Any:
        """Select model using AdaptiveModelRouter if available."""
        if self.adaptive_router:
            return self.adaptive_router.select(
                required_capabilities=capabilities, role=role, task_id=task_id
            )
        return self.router.select(required_capabilities=capabilities, role=role)

    def _infer_capabilities(self, task: Any) -> List[str]:
        caps = ["reasoning"]
        if task.spec.objective:
            caps.append("task_analysis")
        if task.spec.acceptance_criteria:
            caps.append("verification_planning")
        return caps

    def _save_artifact(self, task_id: str, name: str, data: Any):
        """Save an artifact to the execution directory."""
        path = os.path.join(self._artifacts_dir, task_id, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if isinstance(data, (dict, list)):
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                f.write(str(data))

    def _state_to_dict(self, state: Any) -> dict:
        return {
            "task_id": state.task_id,
            "current_state": state.current_state,
            "history": [{"from": h.from_state, "to": h.to_state, "timestamp": h.timestamp}
                       for h in state.history],
        }

    def _build_report(self, task_id: str, state: Any, eval_result: Any, evidence: Any) -> dict:
        report = {"task_id": task_id, "status": state.current_state}
        if eval_result:
            report["evaluation"] = {
                "score": eval_result.score,
                "decision": eval_result.decision,
                "acceptance": [{"criterion": a.criterion, "status": a.status}
                              for a in eval_result.acceptance],
            }
        if evidence:
            report["evidence"] = evidence.to_dict() if hasattr(evidence, 'to_dict') else {}
        report["completed_at"] = datetime.utcnow().isoformat()
        return report
