---
name: upstream-friendly-mjlab
description: >-
  Plan and implement mjlab code, task, sensor, training, or API changes that must
  reuse upstream capabilities where possible and remain easy to rebase and
  contribute while keeping project requirements synchronized. Use before editing
  runtime code or shared interfaces, especially for project-specific 3DGS work that
  should preserve upstream defaults and minimize conflicts with the local main
  branch.
---

# Upstream-friendly mjlab development

Treat the local `main` branch as the read-only upstream baseline. Develop on the
current feature branch. Do not switch, update, merge, rebase, reset, or commit either
branch unless the user explicitly requests it.

## Establish the boundary

Before editing runtime code or shared interfaces:

1. Inspect the current branch, worktree, and merge base with `main`.
2. Search the current tree and `main` for equivalent behavior, reusable components,
   configuration hooks, registries, utilities, and tests before proposing a design.
3. Compare the relevant files with `main` and identify existing extension points.
4. Separate changes already on the feature branch from the new task. Preserve both
   user changes and unrelated branch work.
5. Choose the smallest design that leaves behavior unchanged when the new feature is
   not selected.

Do not fetch merely to refresh `main`. The user maintains the local `main` baseline;
report if it is missing or if its relationship to the feature branch is ambiguous.

## Prefer additive integration

In descending order of preference:

1. Configure, compose, or directly reuse functionality already present on `main`.
2. Add a thin project-specific task, configuration, adapter, asset, or test in a
   dedicated module or namespace.
3. Use an existing registry, configuration field, subclass, callback, or composition
   point.
4. Add a small, general-purpose optional extension point whose default exactly
   preserves existing behavior.
5. Modify shared core behavior only when the first four options cannot meet the
   requirement.

Avoid broad refactors, file moves, renames, formatting churn, dependency updates, or
cleanup unrelated to the requested feature. Do not copy, fork, rename, or reimplement
an upstream capability that can be configured, imported, composed, or minimally
extended. Before introducing a new abstraction, state the concrete gap in the
existing `main` implementation that it fills.

When a shared-core edit is necessary, explain why an additive approach is
insufficient before making a material or behavior-changing edit. Keep the public API
backward compatible, append optional parameters where practical, and retain existing
defaults. Ask for direction before a broad compatibility change.

## Keep the contribution separable

- Keep 3DGS-specific policy, task, assets, and experiments out of generic components
  unless the abstraction is useful to upstream mjlab independently of this project.
- Make feature activation explicit. Existing tasks must not create new sensors,
  renderers, assets, dependencies, or runtime overhead by default.
- Put compatibility shims at the project boundary instead of spreading conditions
  through core modules.
- Add focused tests for the new behavior and regression coverage showing that the
  existing path is unchanged when the feature is disabled.
- Update user-facing docs and changelog only for behavior that actually exists. Keep
  research plans and unimplemented milestones clearly labeled as plans.

## Keep requirements synchronized

Every change to implementation code, tests, executable scripts, schemas, or
behavior-bearing configuration/assets must update the relevant file under
`requirements/` in the same task. This is a completion requirement, not an optional
documentation follow-up.

- Before editing, identify the requirement, milestone, or TODO section that owns the
  behavior. If none exists, add a narrowly scoped entry before treating the code as
  complete.
- Record what was implemented, what was verified, and what remains. Mark an item
  complete only when its stated acceptance evidence exists.
- Keep project-level status tables, detailed checklists, validation evidence, and
  dated update records mutually consistent when the selected requirements document
  uses those sections.
- Do not describe planned behavior as implemented, and do not silently broaden a
  requirement to match an incomplete implementation.
- Before handoff, inspect the changed paths. If code or behavior-bearing files
  changed but no corresponding `requirements/` file changed, the task is incomplete.

Documentation-only work that does not change code or behavior may update
`requirements/` only when it changes project scope, status, evidence, or a technical
decision.

## Verify and hand off

Use the repository's required `uv run`/Makefile checks in proportion to the change.
Before handoff:

1. Run `git diff --check` and the focused tests.
2. Confirm every code or behavior change has its matching `requirements/` update.
3. Inspect the diff against both the initial worktree and `main`.
4. List every modified shared/core file and justify why it had to change.
5. State how existing behavior remains unchanged and identify any remaining rebase or
   upstream-compatibility risk.

Do not claim upstream readiness merely because tests pass. A contribution-ready
change must also be narrowly scoped, default-preserving, documented, and free of
project-only assumptions in shared APIs.
