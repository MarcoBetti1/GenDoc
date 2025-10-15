# GenDoc Prototype Output

## Project Goal

Project goal: Provide clear documentation for Task, TaskList and bootstrap_demo_list in main.py, highlighting how the pieces collaborate.

## main.py::Task

main.py::Task defines the Task class. It marks a task as completed by toggling its flag.

## main.py::TaskList

main.py::TaskList defines the TaskList class. Instances track _tasks. It filters tasks flagged as completed. It collects tasks that are still pending. It computes the fraction of tasks that are complete. It appends a new task to the internal list and returns it.

## main.py::bootstrap_demo_list

main.py::bootstrap_demo_list implements the function bootstrap_demo_list(). It returns TaskList. It seeds a demo list and marks some tasks complete for examples.

## main.py::Task::mark_done

main.py::Task::mark_done implements the method mark_done(self). It marks a task as completed by toggling its flag.

## main.py::TaskList::__init__

main.py::TaskList::__init__ implements the method __init__(self, tasks). It accepts tasks.

## main.py::TaskList::add

main.py::TaskList::add implements the method add(self, title). It returns Task. It appends a new task to the internal list and returns it.

## main.py::TaskList::pending

main.py::TaskList::pending implements the method pending(self). It returns list[Task]. It collects tasks that are still pending.

## main.py::TaskList::completed

main.py::TaskList::completed implements the method completed(self). It returns list[Task]. It filters tasks flagged as completed.

## main.py::TaskList::completion_ratio

main.py::TaskList::completion_ratio implements the method completion_ratio(self). It computes the fraction of tasks that are complete.

## Project Overview

This overview weaves together 9 focused summaries across main.py to support the goal of Provide clear documentation for Task, TaskList and bootstrap_demo_list in main.py, highlighting how the pieces collaborate.
