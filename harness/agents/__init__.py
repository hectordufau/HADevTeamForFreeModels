# harness/agents/__init__.py — Agent Definition Loader
"""
Loads and validates SOUL.md and PROFILE.yaml agent definitions.
"""

import os
import yaml
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field


class AgentError(Exception):
    """Raised on agent definition errors."""


@dataclass
class AgentProfile:
    name: str
    capabilities: List[str] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    autonomy: Dict[str, Any] = field(default_factory=lambda: {"level": 2})


@dataclass
class AgentDefinition:
    role: str
    soul_content: str
    profile: AgentProfile

    def required_capabilities(self) -> List[str]:
        return self.profile.capabilities


class AgentLoader:
    """Discovers and loads agent definitions from the agents/ directory."""

    def __init__(self, agents_dir: str):
        self.agents_dir = agents_dir

    def list_roles(self) -> List[str]:
        """List available agent roles by scanning subdirectories."""
        if not os.path.exists(self.agents_dir):
            return []
        return [
            d for d in os.listdir(self.agents_dir)
            if os.path.isdir(os.path.join(self.agents_dir, d))
            and not d.startswith("_")
        ]

    def load(self, role: str) -> AgentDefinition:
        """Load a specific agent role (SOUL.md + PROFILE.yaml)."""
        role_dir = os.path.join(self.agents_dir, role)
        if not os.path.isdir(role_dir):
            raise AgentError(f"Agent role not found: {role}")

        soul_path = os.path.join(role_dir, "SOUL.md")
        profile_path = os.path.join(role_dir, "PROFILE.yaml")

        soul_content = ""
        if os.path.exists(soul_path):
            with open(soul_path) as f:
                soul_content = f.read()

        if not os.path.exists(profile_path):
            raise AgentError(f"Missing PROFILE.yaml for role: {role}")

        with open(profile_path) as f:
            try:
                profile_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise AgentError(f"Invalid PROFILE.yaml for {role}: {e}")

        if not isinstance(profile_data, dict):
            raise AgentError(f"PROFILE.yaml for {role} must contain a mapping")

        profile = AgentProfile(
            name=profile_data.get("name", role),
            capabilities=profile_data.get("capabilities", []),
            skills=profile_data.get("skills", []),
            tools=profile_data.get("tools", []),
            autonomy=profile_data.get("autonomy", {"level": 2}),
        )

        return AgentDefinition(role=role, soul_content=soul_content, profile=profile)
