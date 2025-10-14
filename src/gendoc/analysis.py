from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from .config import GenDocConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CodeElement:
    """Represents a class, function, or method extracted from source code."""

    identifier: str
    kind: str
    file_path: Path
    start_line: int
    end_line: int
    source: str
    docstring: Optional[str]
    dependencies: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ProjectStructure:
    """Collection of code elements and structural metadata."""

    root: Path
    files: List[Path]
    elements: List[CodeElement]
    tree_repr: str


class ProjectAnalyzer:
    """Extracts structural information from a repository."""

    def __init__(self, config: GenDocConfig) -> None:
        self._config = config

    def analyze(self) -> ProjectStructure:
        python_files = self._discover_python_files(self._config.repo_path)
        elements: list[CodeElement] = []

        for path in python_files:
            try:
                elements.extend(self._extract_elements(path))
            except SyntaxError as exc:
                logger.warning("Failed to parse %s: %s", path, exc)

        tree_repr = self._render_tree(python_files)
        logger.debug("Discovered %d code elements", len(elements))
        return ProjectStructure(
            root=self._config.repo_path,
            files=python_files,
            elements=elements,
            tree_repr=tree_repr,
        )

    def _discover_python_files(self, root: Path) -> List[Path]:
        ignore_dirs = {".git", "venv", "__pycache__", ".mypy_cache"}
        files: list[Path] = []
        for path in root.rglob("*.py"):
            if any(part in ignore_dirs for part in path.parts):
                continue
            files.append(path)
        return sorted(files)

    def _extract_elements(self, file_path: Path) -> List[CodeElement]:
        text = file_path.read_text(encoding="utf-8")
        module = ast.parse(text)
        annotate_ast_with_parents(module)
        elements: list[CodeElement] = []

        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                identifier = self._build_identifier(file_path, node)
                start_line = getattr(node, "lineno", 1)
                end_line = getattr(node, "end_lineno", start_line)
                source = self._slice_source(text, start_line, end_line)
                docstring = ast.get_docstring(node)
                deps = self._collect_dependencies(node)
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                elements.append(
                    CodeElement(
                        identifier=identifier,
                        kind=kind,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        source=source,
                        docstring=docstring,
                        dependencies=deps,
                    )
                )
        return elements

    def _build_identifier(self, file_path: Path, node: ast.AST) -> str:
        relative = file_path.relative_to(self._config.repo_path)
        parts: list[str] = [str(relative)]
        if isinstance(node, ast.ClassDef):
            parts.append(node.name)
        else:
            parent = getattr(node, "parent", None)
            if parent and isinstance(parent, ast.ClassDef):
                parts.extend([parent.name, node.name])
            else:
                parts.append(node.name)  # type: ignore[attr-defined]
        return "::".join(parts)

    def _slice_source(self, text: str, start: int, end: int) -> str:
        lines = text.splitlines()
        snippet = lines[start - 1 : end]
        return "\n".join(snippet)

    def _collect_dependencies(self, node: ast.AST) -> List[str]:
        deps: set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if isinstance(func, ast.Name):
                    deps.add(func.id)
                elif isinstance(func, ast.Attribute):
                    deps.add(func.attr)
        return sorted(deps)

    def _render_tree(self, files: Iterable[Path]) -> str:
        lines: list[str] = []
        root = self._config.repo_path
        for file_path in files:
            relative = file_path.relative_to(root)
            indent = "  " * (len(relative.parts) - 1)
            lines.append(f"{indent}- {relative}")
        return "\n".join(lines)


def annotate_ast_with_parents(module: ast.AST) -> None:
    """Augment AST nodes with parent references."""

    for node in ast.walk(module):
        for child in ast.iter_child_nodes(node):
            setattr(child, "parent", node)
