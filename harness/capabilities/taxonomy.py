# harness/capabilities/taxonomy.py — CapabilityTaxonomy (V2.2)
"""
Loads and queries the capability hierarchy from config/capability_taxonomy.yaml.
Supports parent/child/sibling/ancestor navigation.
"""

from typing import Any, Dict, List, Optional, Set
import os
import yaml


class CapabilityTaxonomyError(Exception):
    """Raised on taxonomy errors."""


class CapabilityTaxonomy:
    """Loads and queries the capability hierarchy."""

    def __init__(self, taxonomy_path: Optional[str] = None):
        self._tree: dict = {}
        # Internal lookup maps
        self._children: Dict[str, List[str]] = {}  # parent -> children
        self._parents: Dict[str, List[str]] = {}   # child -> parents
        self._dependencies: Dict[str, List[str]] = {}  # capability -> dependencies
        self._all_capabilities: Set[str] = set()

        if taxonomy_path is None:
            # Walk up from harness/capabilities/ -> config/
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            taxonomy_path = os.path.join(base, "config", "capability_taxonomy.yaml")

        if os.path.exists(taxonomy_path):
            self.load(taxonomy_path)

    def load(self, path: str):
        """Load taxonomy from YAML file."""
        if not os.path.exists(path):
            raise CapabilityTaxonomyError(f"Taxonomy file not found: {path}")

        with open(path) as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise CapabilityTaxonomyError(f"Invalid YAML in {path}: {e}")

        if not isinstance(data, dict):
            raise CapabilityTaxonomyError("Taxonomy must be a mapping")

        self._tree = data
        self._build_index(self._tree.get("engineering", {}))

        # Load declarative capability dependencies
        deps = data.get("capability_dependencies", {})
        if isinstance(deps, dict):
            self._dependencies = {
                cap: list(dep_list) if isinstance(dep_list, list) else []
                for cap, dep_list in deps.items()
            }

    def _build_index(self, subtree: dict, parent_path: Optional[str] = None):
        """Recursively build parent/child index from tree."""
        for name, children in subtree.items():
            self._all_capabilities.add(name)
            if parent_path:
                self._parents.setdefault(name, []).append(parent_path)
                self._children.setdefault(parent_path, []).append(name)

            if isinstance(children, dict) and children:
                self._build_index(children, name)

    def get_children(self, capability: str) -> List[str]:
        """Get more specific capabilities under this one."""
        return self._children.get(capability, [])

    def get_parents(self, capability: str) -> List[str]:
        """Get broader capabilities that encompass this one."""
        return self._parents.get(capability, [])

    def get_siblings(self, capability: str) -> List[str]:
        """Get same-level capabilities under the same parent."""
        parents = self.get_parents(capability)
        siblings = set()
        for parent in parents:
            for child in self._children.get(parent, []):
                if child != capability:
                    siblings.add(child)
        return sorted(siblings)

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        """Check if capability A is an ancestor of capability B."""
        if ancestor == descendant:
            return False
        visited = set()
        queue = [descendant]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            parents = self.get_parents(current)
            if ancestor in parents:
                return True
            queue.extend(parents)
        return False

    def is_descendant(self, descendant: str, ancestor: str) -> bool:
        """Check if capability B is a descendant of capability A."""
        return self.is_ancestor(ancestor, descendant)

    def get_all_capabilities(self) -> Set[str]:
        """Return the set of all known capabilities."""
        return self._all_capabilities

    def get_leaves(self) -> List[str]:
        """Get leaf capabilities (those with no children)."""
        return [c for c in self._all_capabilities if not self._children.get(c)]

    def capability_exists(self, capability: str) -> bool:
        """Check if a capability is defined in the taxonomy."""
        return capability in self._all_capabilities

    def get_dependencies(self, capability: str) -> List[str]:
        """Get the declared dependencies of a capability (from capabilility_dependencies config)."""
        return list(self._dependencies.get(capability, []))

    def has_dependencies_declared(self, capability: str) -> bool:
        """Check if a capability has dependency declarations in config."""
        return capability in self._dependencies
