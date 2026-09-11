"""
Model Performance Registry and Adaptive Model Router (V2.1).
"""

import os
import json
import sqlite3
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelTaskRecord:
    task_id: str
    model_id: str
    role: str
    success: bool
    score: float
    iterations: int
    latency_ms: float
    capabilities_used: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class ModelPerformanceRegistry:
    """Persistent store for model performance metrics using SQLite."""

    def __init__(self, db_path: str = ""):
        if not db_path:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "artifacts", "model_performance.db"
            )
        self.db_path = db_path
        self._record_type = ModelTaskRecord
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    score REAL NOT NULL DEFAULT 0.0,
                    iterations INTEGER NOT NULL DEFAULT 1,
                    latency_ms REAL NOT NULL DEFAULT 0.0,
                    capabilities_used TEXT DEFAULT '[]',
                    timestamp TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_stats
                ON model_performance(model_id, role)
            """)
            conn.commit()

    def record(self, record: ModelTaskRecord):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO model_performance
                (task_id, model_id, role, success, score, iterations, latency_ms, capabilities_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.task_id, record.model_id, record.role,
                1 if record.success else 0, record.score,
                record.iterations, record.latency_ms,
                json.dumps(record.capabilities_used),
                record.timestamp
            ))
            conn.commit()

    def get_model_stats(self, model_id: str = None, role: str = None) -> List[dict]:
        query = ("SELECT model_id, role, COUNT(*) as total, SUM(success) as successful, "
                 "AVG(score) as avg_score, AVG(iterations) as avg_iterations, "
                 "AVG(latency_ms) as avg_latency FROM model_performance")
        params = []
        conditions = []
        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)
        if role:
            conditions.append("role = ?")
            params.append(role)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY model_id, role"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_best_model_for_capability(self, capability: str, min_records: int = 3) -> Optional[dict]:
        """Find the model with highest success rate for a given capability."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT model_id, COUNT(*) as total, SUM(success) as successful,
                       AVG(score) as avg_score, AVG(iterations) as avg_iterations
                FROM model_performance
                WHERE capabilities_used LIKE ?
                GROUP BY model_id
                HAVING total >= ?
                ORDER BY CAST(SUM(success) AS REAL) / COUNT(*) DESC, avg_score DESC
                LIMIT 1
            """, (f'%{capability}%', min_records)).fetchall()
            return dict(rows[0]) if rows else None

    def get_capability_leaderboard(self, min_records: int = 3) -> List[dict]:
        """Get success rate per model per capability."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT model_id, capabilities_used,
                       COUNT(*) as total, SUM(success) as successful,
                       AVG(score) as avg_score
                FROM model_performance
                GROUP BY model_id, capabilities_used
                HAVING total >= ?
                ORDER BY successful * 1.0 / total DESC
            """, (min_records,)).fetchall()
            return [dict(r) for r in rows]


class AdaptiveModelRouter:
    """Model router that learns from historical performance."""

    def __init__(self, catalog: Any, registry: ModelPerformanceRegistry, fallback_router: Any):
        self.catalog = catalog
        self.registry = registry
        self.fallback = fallback_router

    def select(self, required_capabilities: List[str], role: str = "", task_id: str = "",
               prefer_fallback: bool = False) -> Any:
        """
        Select model using historical data when available.
        Falls back to capability-based ranking.
        """
        from harness.routing import ModelSelection, RouterError

        candidates = self.catalog.list_free()
        if not candidates:
            raise RouterError("No free models available")

        scored = []
        # Variables for the reason string (last iteration values)
        last_cap = 0.0
        last_hist = 0.0
        for model in candidates:
            # Base score from capability match (normalized 0-1)
            num_caps = len(required_capabilities)
            cap_score = sum(model.capabilities.get(c, 0.0) for c in required_capabilities)
            if num_caps > 0:
                cap_score /= num_caps  # normalize to 0-1

            # Historical bonus (success rate weighted 30%)
            stats = self.registry.get_model_stats(model_id=model.model_id, role=role)
            hist_bonus = 0.0
            if stats:
                s = stats[0]
                success_rate = s["successful"] / s["total"] if s["total"] > 0 else 0
                hist_bonus = success_rate * 0.3  # 30% weight

            total = cap_score + hist_bonus
            if cap_score > 0:
                scored.append((total, model))
                last_cap = cap_score
                last_hist = hist_bonus

        if not scored:
            return self.fallback.select(required_capabilities, role, prefer_fallback)

        scored.sort(key=lambda x: x[0], reverse=True)
        idx = 1 if prefer_fallback and len(scored) > 1 else 0
        _, best = scored[idx]

        return ModelSelection(
            model_id=best.model_id,
            provider=best.provider,
            cost=best.cost,
            reason=f"Adaptive: {best.model_id} score={scored[idx][0]:.2f} (cap={last_cap:.2f}, hist={last_hist:.2f})",
            fallback_used=idx > 0,
        )
