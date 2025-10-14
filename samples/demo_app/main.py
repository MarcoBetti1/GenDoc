"""Demo application for GenDoc prototype.

The goal is to provide a compact yet non-trivial structure that exercises
function, class, and helper extraction logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List


@dataclass(slots=True)
class Task:
    """Represents a todo item with status metadata."""

    title: str
    completed: bool = False
    created_at: datetime = datetime.utcnow()

    def mark_done(self) -> None:
        """Mark the task as completed."""
        self.completed = True


class TaskList:
    """Simple in-memory task list with statistics helpers."""

    def __init__(self, tasks: Iterable[Task] | None = None) -> None:
        self._tasks: List[Task] = list(tasks or [])

    def add(self, title: str) -> Task:
        task = Task(title=title)
        self._tasks.append(task)
        return task

    def pending(self) -> list[Task]:
        return [task for task in self._tasks if not task.completed]

    def completed(self) -> list[Task]:
        return [task for task in self._tasks if task.completed]

    def completion_ratio(self) -> float:
        total = len(self._tasks)
        if total == 0:
            return 0.0
        return len(self.completed()) / total


def bootstrap_demo_list() -> TaskList:
    """Create a populated task list for demo output."""
    tl = TaskList()
    tl.add("draft design doc")
    tl.add("wire CLI skeleton")
    third = tl.add("hook up OpenAI client")
    third.mark_done()
    return tl


if __name__ == "__main__":
    task_list = bootstrap_demo_list()
    ratio = task_list.completion_ratio()
    print(f"Demo task list is {ratio:.0%} complete")
