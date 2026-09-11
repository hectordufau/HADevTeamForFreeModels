# harness/orchestrator/__init__.py — Orchestrator and End-to-End Harness
"""
The Orchestrator is the central coordination component.
Controls the complete task lifecycle from intake to delivery.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import os
import yaml


class OrchestratorError(Exception):
    """Raised on orchestration failures."""


class Orchestrator:
    """Coordinates the complete task lifecycle."""

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
        self._artifacts_dir = "artifacts/executions"

    def run(self, task_id: str) -> Dict[str, Any]:
        """
        Execute a complete task lifecycle.

        Returns the final execution record.
        """
        # Load task contract
        task = self.tasks.load(os.path.join("tasks", "active", f"{task_id}.yaml"))

        # Register and transition to ANALYZING
        state = self.state.register(task_id)
        state.transition("ANALYZING")
        self._save_state(task_id)

        # Build initial context
        context_pack = self.context.build_context(
            task_contract=task,
            agent_role="manager",
        )

        # Determine required capabilities from task
        required_capabilities = self._infer_capabilities(task)

        # Select manager model
        manager_selection = self.router.select(
            required_capabilities=required_capabilities,
            role="manager",
        )

        # Load manager agent
        manager_agent = self.agents.load("manager")

        # Execute manager (analysis phase)
        manager_result = self.executor.execute(
            agent_def=manager_agent,
            model_id=manager_selection.model_id,
            task_contract=task,
            context_pack=context_pack,
        )

        if manager_result.status == "failed":
            state.transition("FAILED")
            self._save_state(task_id)
            return {"task_id": task_id, "status": "FAILED", "error": manager_result.error}

        # Transition to PLANNED
        state.transition("PLANNED")
        self._save_state(task_id)

        # --- Architecture phase ---
        arch_agent = self.agents.load("architect")
        arch_selection = self.router.select(
            required_capabilities=["architecture", "reasoning", "code_analysis"],
            role="architect",
        )
        arch_context = self.context.build_context(
            task_contract=task,
            agent_role="architect",
        )
        arch_result = self.executor.execute(arch_agent, arch_selection.model_id, task, arch_context)
        if arch_result.status == "failed":
            state.transition("FAILED")
            self._save_state(task_id)
            return {"task_id": task_id, "status": "FAILED", "error": arch_result.error}

        # --- Implementation phase ---
        state.transition("IMPLEMENTING")
        self._save_state(task_id)

        coder_agent = self.agents.load("coder")
        coder_selection = self.router.select(
            required_capabilities=["coding", "debugging"],
            role="coder",
        )
        coder_context = self.context.build_context(
            task_contract=task,
            agent_role="coder",
            relevant_files=task.spec.allowed_changes,
        )
        coder_result = self.executor.execute(coder_agent, coder_selection.model_id, task, coder_context)

        if coder_result.status == "failed":
            state.transition("FAILED")
            self._save_state(task_id)
            return {"task_id": task_id, "status": "FAILED", "error": coder_result.error}

        # --- Verification phase ---
        state.transition("VERIFYING")
        self._save_state(task_id)

        verif_result = self.verifier.verify(
            task, workspace=".", checks=task.spec.verification.required
        )

        # --- Evidence collection ---
        evidence_record = self.evidence.collect(coder_result, task_id)
        self.evidence.persist(evidence_record)

        # --- Evaluation ---
        state.transition("EVALUATING")
        self._save_state(task_id)

        eval_result = self.evaluator.evaluate(task, verif_result, evidence_record)

        # --- Iteration or Review ---
        iteration = 1
        while not eval_result.is_passed() and iteration <= self.config.get("execution", {}).get("max_iterations", 3):
            # Build feedback
            feedback = self.iteration.build_feedback(task, eval_result, verif_result)

            if not self.iteration.should_iterate(iteration, feedback):
                state.transition("NEEDS_HUMAN" if feedback.failures else "COMPLETED")
                self._save_state(task_id)
                break

            state.transition("ITERATING")
            self._save_state(task_id)

            # Re-execute coder with feedback
            coder_context.memory_notes.append(f"Iteration {iteration}: {feedback.recommendations}")
            coder_result = self.executor.execute(coder_agent, coder_selection.model_id, task, coder_context)

            # Re-verify
            verif_result = self.verifier.verify(
                task, workspace=".", checks=task.spec.verification.required
            )
            evidence_record = self.evidence.collect(coder_result, task_id)
            eval_result = self.evaluator.evaluate(task, verif_result, evidence_record)
            iteration += 1

        # --- Review phase ---
        if state.current_state not in {"FAILED", "CANCELLED", "NEEDS_HUMAN", "COMPLETED", "BLOCKED"}:
            state.transition("REVIEWING")
            self._save_state(task_id)

            # Reviewer evaluates final result
            reviewer_agent = self.agents.load("reviewer")
            reviewer_selection = self.router.select(
                required_capabilities=["code_review", "reasoning", "security"],
                role="reviewer",
            )

            if eval_result.is_passed():
                state.transition("COMPLETED")
            else:
                state.transition("FAILED")

            self._save_state(task_id)

        # --- Final report ---
        report = self._build_report(task_id, state, eval_result, evidence_record)
        self._save_report(task_id, report)

        return report

    def _infer_capabilities(self, task: Any) -> List[str]:
        """Infer required capabilities from a task contract."""
        caps = ["reasoning"]
        spec = task.spec

        if spec.objective:
            caps.append("task_analysis")

        if spec.acceptance_criteria:
            caps.append("verification_planning")

        return caps

    def _save_state(self, task_id: str):
        """Persist current state to disk."""
        state = self.state.get(task_id)
        if state:
            state_path = os.path.join(self._artifacts_dir, task_id, "state.yaml")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            self.state.persist(task_id, state_path)

    def _build_report(self, task_id: str, state: Any, eval_result: Any, evidence: Any) -> dict:
        """Build the final execution report."""
        return {
            "task_id": task_id,
            "status": state.current_state,
            "evaluation": {
                "score": eval_result.score,
                "decision": eval_result.decision,
                "acceptance": [{"criterion": a.criterion, "status": a.status} for a in eval_result.acceptance],
            },
            "evidence": evidence.to_dict() if hasattr(evidence, 'to_dict') else {},
            "completed_at": datetime.utcnow().isoformat(),
        }

    def _save_report(self, task_id: str, report: dict):
        """Persist the final execution report."""
        report_path = os.path.join(self._artifacts_dir, task_id, "final_report.yaml")
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
