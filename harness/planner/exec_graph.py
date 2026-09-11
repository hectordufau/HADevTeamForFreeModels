# harness/planner/exec_graph.py — ExecutionGraph (V2.2)
"""
Parallel-safe DAG runner.
Executes nodes in dependency order; independent nodes run concurrently.
"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


class ExecutionGraphError(Exception):
    """Raised on execution graph errors."""


@dataclass
class NodeExecutionResult:
    """Result of executing a single workflow node."""
    node_id: str
    capability: str
    agent: str
    status: str  # completed, failed, skipped
    result: Optional[dict] = None
    error: Optional[str] = None
    evidence: Optional[dict] = None


class ExecutionGraph:
    """Executes a workflow DAG with parallel nodes."""

    def __init__(self, workflow: Any, orchestrator: Any):
        self.workflow = workflow
        self.orch = orchestrator
        self._results: Dict[str, NodeExecutionResult] = {}
        self._completed: Set[str] = set()
        self._failed: Set[str] = set()

    async def execute(self) -> Dict[str, Any]:
        """
        Execute nodes in dependency order.
        Independent nodes run in parallel.
        Returns aggregated results.
        """
        # Build dependency map
        node_map: Dict[str, Any] = {}
        dep_map: Dict[str, Set[str]] = {}

        nodes = self.workflow.nodes if hasattr(self.workflow, 'nodes') else self.workflow.get("nodes", [])

        for node in nodes:
            nid = node.id if hasattr(node, 'id') else node.get("id", "")
            node_map[nid] = node
            deps = node.depends_on if hasattr(node, 'depends_on') else node.get("depends_on", [])
            dep_map[nid] = set(deps)

        total = len(node_map)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures_map: Dict[Any, str] = {}

            while len(self._completed) + len(self._failed) < total:
                # Find ready nodes
                ready_nodes = []
                for nid, deps in dep_map.items():
                    if nid in self._completed or nid in self._failed or nid in futures_map.values():
                        continue
                    if all(d in self._completed for d in deps):
                        ready_nodes.append(nid)

                # Submit ready nodes
                for nid in ready_nodes:
                    future = executor.submit(self._execute_node, node_map[nid])
                    futures_map[future] = nid

                if not futures_map:
                    # No running futures and no ready nodes — check for stalled
                    for nid, deps in dep_map.items():
                        if nid not in self._completed and nid not in self._failed:
                            if any(d in self._failed for d in deps):
                                self._failed.add(nid)
                                self._results[nid] = NodeExecutionResult(
                                    node_id=nid,
                                    capability=node_map[nid].capability if hasattr(node_map[nid], 'capability') else node_map[nid].get("capability", ""),
                                    agent=node_map[nid].agent if hasattr(node_map[nid], 'agent') else node_map[nid].get("agent", ""),
                                    status="skipped",
                                    result={"reason": "Dependency failed"},
                                )
                    if not self._failed:
                        break
                    continue

                # Wait for at least one future to complete
                done, _ = wait(list(futures_map.keys()), return_when=FIRST_COMPLETED)

                for future in done:
                    nid = futures_map.pop(future)
                    try:
                        nresult = future.result()
                        self._results[nid] = nresult
                        if nresult.status == "completed":
                            self._completed.add(nid)
                        else:
                            self._failed.add(nid)
                    except Exception as e:
                        self._failed.add(nid)
                        self._results[nid] = NodeExecutionResult(
                            node_id=nid,
                            capability="",
                            agent="",
                            status="failed",
                            error=str(e),
                        )

        return self._aggregate_results()

    def _execute_node(self, node: Any) -> NodeExecutionResult:
        """Execute a single workflow node."""
        nid = node.id if hasattr(node, 'id') else node.get("id", "")
        ncap = node.capability if hasattr(node, 'capability') else node.get("capability", "")
        nagent = node.agent if hasattr(node, 'agent') else node.get("agent", "")
        nmodel = node.model_id if hasattr(node, 'model_id') else node.get("model_id", "")

        try:
            # Load agent
            agent_def = self.orch.agents.load(nagent)

            # Build context
            task = None
            if hasattr(self.orch, '_current_task'):
                task = self.orch._current_task

            context = self.orch.context.build_context(
                task_contract=task,
                agent_role=nagent,
                agent_soul=agent_def.soul_content,
                relevant_files=[],
            )

            # Execute
            result = self.orch.executor.execute(
                agent_def=agent_def,
                model_id=nmodel,
                task_contract=task,
                context_pack=context,
            )

            # Collect evidence
            evidence = None
            if hasattr(self.orch, 'evidence'):
                ev = self.orch.evidence.collect(result, task.metadata.id if task else "")
                evidence = ev.to_dict() if hasattr(ev, 'to_dict') else {}

            return NodeExecutionResult(
                node_id=nid,
                capability=ncap,
                agent=nagent,
                status=result.status,
                result={"changes": result.changes, "claims": result.claims}
                    if hasattr(result, 'changes') else {},
                evidence=evidence,
            )

        except Exception as e:
            return NodeExecutionResult(
                node_id=nid,
                capability=ncap,
                agent=nagent,
                status="failed",
                error=str(e),
            )

    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate all node results into a final report."""
        total = len(self._completed) + len(self._failed)
        failed_count = len(self._failed)

        return {
            "status": "completed" if failed_count == 0 else "failed",
            "completed_nodes": sorted(self._completed),
            "failed_nodes": sorted(self._failed),
            "total_nodes": total,
            "node_results": {
                nid: {
                    "status": r.status,
                    "capability": r.capability,
                    "agent": r.agent,
                    "error": r.error,
                }
                for nid, r in self._results.items()
            },
        }
