# harness/planner/observability.py — Observability (Phase L)
"""
Structured logging, decision tracing, and execution trace for the V3 harness.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
import threading


@dataclass
class LogEntry:
    """A structured log entry with context."""
    level: str  # DEBUG, INFO, WARNING, ERROR
    module: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "module": self.module,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }


@dataclass
class DecisionRecord:
    """Record of a decision made during execution."""
    decision_id: str
    decision_type: str  # agent_selection, model_selection, plan_choice, branch_resolution, replan
    context: Dict[str, Any]
    alternatives: List[Dict[str, Any]]
    selected: str
    rationale: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    trace_id: str = ""

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type,
            "context": self.context,
            "alternatives": self.alternatives,
            "selected": self.selected,
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
        }


@dataclass
class ExecutionTraceNode:
    """A node in the execution trace tree."""
    node_id: str
    component: str
    action: str
    duration_ms: float
    status: str  # success, failure, skipped
    input_summary: str = ""
    output_summary: str = ""
    children: List['ExecutionTraceNode'] = field(default_factory=list)
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "component": self.component,
            "action": self.action,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "input_summary": self.input_summary[:200],
            "output_summary": self.output_summary[:200],
            "children": [c.to_dict() for c in self.children],
            "error": self.error,
            "timestamp": self.timestamp,
        }


class StructuredLogger:
    """Logger that produces structured log entries."""

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "logs"
        )
        self._lock = threading.Lock()
        os.makedirs(self.storage_dir, exist_ok=True)
        self._entries: List[LogEntry] = []

    def log(self, level: str, module: str, message: str,
            context: Optional[Dict[str, Any]] = None,
            trace_id: str = ""):
        """Log a structured entry."""
        entry = LogEntry(
            level=level,
            module=module,
            message=message,
            context=context or {},
            trace_id=trace_id,
        )
        with self._lock:
            self._entries.append(entry)

    def info(self, module: str, message: str, **kwargs):
        self.log("INFO", module, message, **kwargs)

    def warning(self, module: str, message: str, **kwargs):
        self.log("WARNING", module, message, **kwargs)

    def error(self, module: str, message: str, **kwargs):
        self.log("ERROR", module, message, **kwargs)

    def debug(self, module: str, message: str, **kwargs):
        self.log("DEBUG", module, message, **kwargs)

    def get_recent(self, level: str = "", limit: int = 100) -> List[LogEntry]:
        """Get recent log entries, optionally filtered by level."""
        entries = self._entries
        if level:
            entries = [e for e in entries if e.level == level]
        return entries[-limit:]

    def flush(self, filename: str = ""):
        """Flush log entries to disk."""
        filename = filename or f"log_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        path = os.path.join(self.storage_dir, filename)
        with self._lock:
            with open(path, "a") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry.to_dict()) + "\n")
            self._entries.clear()


class DecisionTracer:
    """Records and traces decisions made during execution."""

    def __init__(self, storage_dir: str = ""):
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "artifacts", "decisions"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        self._decisions: List[DecisionRecord] = []

    def record(self, decision: DecisionRecord):
        """Record a decision."""
        self._decisions.append(decision)
        self._persist(decision)

    def get_decisions(self, trace_id: str = "") -> List[DecisionRecord]:
        """Get decisions, optionally filtered by trace_id."""
        if trace_id:
            return [d for d in self._decisions if d.trace_id == trace_id]
        return list(self._decisions)

    def _persist(self, decision: DecisionRecord):
        """Persist a decision to disk."""
        path = os.path.join(self.storage_dir, f"{decision.decision_id}.json")
        with open(path, "w") as f:
            json.dump(decision.to_dict(), f, indent=2)


class ExecutionTracer:
    """Builds an execution trace tree for observability."""

    def __init__(self):
        self._current_trace: Optional[ExecutionTraceNode] = None
        self._stack: List[ExecutionTraceNode] = []
        self._root: Optional[ExecutionTraceNode] = None

    def begin(self, component: str, action: str,
              input_summary: str = "") -> str:
        """Begin a new trace node."""
        node_id = f"{component}_{action}_{datetime.utcnow().strftime('%H%M%S%f')}"
        node = ExecutionTraceNode(
            node_id=node_id,
            component=component,
            action=action,
            duration_ms=0.0,
            status="running",
            input_summary=input_summary,
        )

        if self._stack:
            self._stack[-1].children.append(node)
        elif not self._root:
            self._root = node

        self._stack.append(node)
        self._current_trace = node
        return node_id

    def end(self, node_id: str, status: str = "success",
            output_summary: str = "", error: Optional[str] = None):
        """End a trace node."""
        if not self._stack:
            return
        node = self._stack.pop()
        if node.node_id == node_id:
            node.status = status
            node.output_summary = output_summary
            node.error = error
            if self._stack:
                self._current_trace = self._stack[-1]
            else:
                self._current_trace = self._root

    def get_trace(self) -> Optional[ExecutionTraceNode]:
        """Get the full execution trace tree."""
        return self._root

    def reset(self):
        """Reset the trace."""
        self._current_trace = None
        self._stack.clear()
        self._root = None
