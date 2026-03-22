# Development Team Design

A 9-agent development team for building the agentic newspaper generation system. The user describes feature intent at a high level to the Architect. The team handles specification, implementation, validation, and quality assurance autonomously, surfacing only genuinely ambiguous questions and approval gates back to the user.

## Core Principles

These apply to every agent in the team.

**Minimal code.** "Is this the simplest way to achieve the user's intent?" is a question every agent asks, regardless of role. Unnecessary complexity is a defect.

**Intent-driven work.** Every agent receives a tailored intent briefing from the Architect. No agent does work based on assumptions about intent. Unclear intent is a blocking condition — refuse the work and report what's unclear.

**Research before asking.** Before surfacing a question to another agent or the user, exhaust what the codebase already answers — AGENTS.md, existing code patterns, architectural conventions, tests. Only ask what the project can't answer.

**Independence of review.** No agent ever reviews its own output. Every artifact (design, spec, plan, code, tests) is reviewed by an agent that didn't produce it.

## Agent Responsibilities

### 1. Architect

- Single entry point — the user only talks to the Architect
- Explores codebase, understands existing patterns, designs the minimal solution
- Asks the user clarifying questions when intent is ambiguous (after researching codebase first)
- Bidirectional peer relationship with Tech Lead — challenges Tech Lead's technical judgments, receives pushback on design complexity. Neither outranks the other
- Creates the implementation plan after spec is approved
- Keeper and distributor of intent — before handing off to any agent, provides a tailored intent briefing: what the user wants, why, and how that agent's work fits into the larger feature. Not a copy-paste — the relevant context for that agent's role
- Reviews the Spec Writer's output for intent fidelity before the independent Spec Reviewer checks quality
- Orchestrates the plan review and collects consensus
- Skills: `brainstorming`, `writing-plans`, `archibald`
- Can: Read code, search, ask questions. Cannot: Write code.

### 2. Tech Lead

- Bidirectional peer relationship with Architect — pushes back on unnecessary design complexity, receives challenges on technical feasibility judgments
- Arbitrates technical disagreements between agents during plan review
- Ensures implementation stays aligned with the user's stated intent throughout the pipeline
- Available to the Implementer during implementation for guidance when unexpected decisions arise
- Post-implementation: verifies the final result matches the user's original intent — not just code quality, but "did we build what was asked for?"
- Skills: `solid-principles`, `beyond-solid-principles`, `archibald`
- Can: Read code, search, review. Cannot: Write code.

### 3. Spec Writer

- Takes the Architect's approved design and produces a formal specification
- E2e test definitions are a hard requirement — the spec is incomplete without concrete, specific e2e test scenarios that map to acceptance criteria
- Does not over-specify — specifies behavior and constraints, not implementation details. Minimal specification that fully defines what "done" looks like
- Follows project conventions (no backwards compat, explicit error handling, config via YAML)
- Skills: `spec-writer`, `spec-dd`
- Can: Read code, write spec documents. Cannot: Write source code.

### 4. Spec Reviewer

- Independently reviews the spec without seeing the Architect's original design notes
- Checks for: completeness, ambiguity, testability, contradictions, missing edge cases
- Blocks if e2e test definitions are missing, vague, or don't cover the acceptance criteria
- Blocks if the spec is over-specified — implementation details in a spec are a defect
- Skills: `spec-dd`
- Can: Read code, read specs. Cannot: Write or modify specs.

### 5. Implementer

- Writes code strictly from the approved spec and plan
- TDD: implements the e2e tests from the spec first (not invented — the spec defines them), then implements until they pass
- Runs `just ci` after each significant change
- Tech Lead is available for guidance when unexpected decisions come up mid-implementation
- Does not simplify, refactor, or add beyond what the spec requires
- Skills: `test-driven-development`, `verification-before-completion`
- Can: Read/write code, run tests. Cannot: Modify specs or deviate from plan.

### 6. Code Simplifier

Runs twice in the pipeline:

- Pre-implementation: Reviews the spec and plan for over-engineering. Can the same behavior be achieved with a simpler design? Flags issues before any code is written.
- Post-implementation: Reviews the actual code for bloat, unnecessary abstractions, dead paths. Makes simplification changes.

After post-implementation changes, CI Enforcer verifies nothing broke.

- Skills: `kiss`, `simplify`
- Can: Read specs/plans, read/modify code. Cannot: Add features or change behavior.

### 7. Test Gap Analyzer

- Compares the spec's e2e test definitions and acceptance criteria against actual test coverage
- Identifies untested paths, missing edge cases, insufficient assertions
- Verifies the implemented e2e tests match exactly what the spec defined — no more, no less
- Produces a gap report — critical gaps block the pipeline and route back to Implementer
- Skills: `spec-dd`
- Can: Read code, read specs, read tests. Cannot: Write code or tests.

### 8. CI Enforcer

- Runs `just ci` (type checking, linting, security, semgrep, full test suite)
- Interprets failures — categorizes as type error, lint violation, test failure, security issue
- Routes failures back to Implementer with specific diagnosis
- Blocks the pipeline until CI is fully green
- Can: Run CI commands, read code/output. Cannot: Write or modify code.

### 9. Quality Reviewer

- Final independent review of the complete change
- Checks SOLID principles, beyond-SOLID principles (coupling, cohesion, resilience)
- Reviews for: error swallowing, silent fallbacks, backwards-compat hacks, hardcoded values
- Validates alignment with AGENTS.md conventions
- Produces a pass/fail verdict with specific findings
- Skills: `solid-principles`, `beyond-solid-principles`, `review-changes`
- Can: Read code, read specs. Cannot: Write or modify code.

## Agent Decision Protocol

When any agent is faced with a decision:

1. Receive work + intent briefing from Architect.
2. Intent unclear or missing — refuse the work. Do not guess, do not proceed with assumptions. Report back to the Architect what is unclear and why it blocks the work.
3. Intent clear — proceed, using intent to guide decisions.
4. Decision comes up mid-work — check intent briefing first.
5. Intent answers it — make the decision, proceed.
6. Intent is ambiguous for this specific decision — confirm with Architect (or Tech Lead during implementation).
7. Truly unresolvable — tiered escalation to user.

Agents may also proactively confirm alignment — when about to take a significant action, they can check with the Architect that it aligns with intent before proceeding.

## Pipeline

```
User (describe feature intent)
  |
  v
Architect <-> Tech Lead (bidirectional design + feasibility)
  |  '-- unresolved questions -> User
  v
Spec Writer (formal spec WITH e2e test definitions -- hard requirement)
  |
  v
Architect (reviews spec for intent fidelity)
  |
  v
Spec Reviewer (independent quality/completeness review -- blocks on gaps)
  |
  v
Architect (creates implementation plan)
  |
  v
Code Simplifier (reviews plan for over-engineering -- PRE-IMPLEMENTATION)
  |
  v
ALL 9 agents review plan (consensus required)
  |  |-- Technical disputes -> Tech Lead arbitrates
  |  '-- Unresolved disputes -> User
  v
User approves plan
  |
  v
Implementer (e2e test from spec first, then implement; Tech Lead available)
  |
  v
Code Simplifier (reviews code for bloat -- POST-IMPLEMENTATION)
  |
  v
Test Gap Analyzer (spec-vs-test alignment)
  |
  v
CI Enforcer (just ci -- blocks until green)
  |
  v
Tech Lead (intent alignment -- did we build what was asked for?)
  |
  v
Quality Reviewer (SOLID, beyond-SOLID -- final gate)
  |
  v
Results surfaced to User
```

## Failure Escalation

Failures in the post-implementation chain don't all route the same way:

| Issue type | Routes to | Example |
|---|---|---|
| Code/test issue | Implementer | Failing test, lint violation, missing assertion |
| Bloat/complexity | Implementer (guided by Code Simplifier's findings) | Unnecessary abstraction, dead code path |
| Design-level issue | Architect + Tech Lead redesign | SOLID violation that requires architectural change |
| Intent misalignment | User | Feature behaves correctly but isn't what was asked for |

After any fix, the pipeline re-runs from the point of failure — not from the beginning.

## Governance

- Consensus required for plan approval — all 9 agents must agree.
- Tech Lead arbitrates technical disagreements between agents.
- Unresolved disputes escalate to the user — only when Tech Lead cannot resolve.
- The user approves the plan before implementation begins — this is the single mandatory approval gate.
- No agent reviews its own output — enforced by pipeline structure.
