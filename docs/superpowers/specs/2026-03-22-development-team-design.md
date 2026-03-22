# Development Team Design

A 9-agent development team for building the agentic newspaper generation system. The user describes feature intent at a high level to the Architect. The team handles specification, implementation, validation, and quality assurance autonomously, surfacing only genuinely ambiguous questions and approval gates back to the user.

## Core Principles

These apply to every agent in the team.

**Minimal code.** "Is this the simplest way to achieve the user's intent?" is a question every agent asks, regardless of role. Unnecessary complexity is a defect.

**Intent-driven work.** Every agent receives a tailored intent briefing from the Architect. No agent does work based on assumptions about intent. Unclear intent is a blocking condition — refuse the work and report what's unclear.

**Research before asking.** Before surfacing a question to another agent or the user, exhaust what the codebase already answers — AGENTS.md, existing code patterns, architectural conventions, tests. Only ask what the project can't answer.

**Independence of review.** No agent reviews its own output. Every artifact (design, spec, plan, code, tests) is reviewed by an agent that didn't produce it. Exception: the Architect participates in the plan review consensus step as the plan's author, presenting and defending the plan rather than reviewing it. The Architect does not cast a review vote on the plan — only the other 8 agents do.

## Agent Responsibilities

### 1. Architect

- Single entry point — the user only talks to the Architect
- Explores codebase, understands existing patterns, designs the minimal solution
- Asks the user clarifying questions when intent is ambiguous (after researching codebase first)
- Bidirectional peer relationship with Tech Lead — challenges Tech Lead's technical judgments, receives pushback on design complexity. Neither outranks the other
- Creates the implementation plan after spec is approved
- Keeper and distributor of intent — before handing off to any agent, provides a tailored intent briefing: what the user wants, why, and how that agent's work fits into the larger feature. Not a copy-paste — the relevant context for that agent's role
- Reviews the Spec Writer's output for intent fidelity before the independent Spec Reviewer checks quality. The Spec Reviewer is unaware that the Architect performed a prior fidelity review — the spec is presented to the Spec Reviewer as-is for independent assessment
- Orchestrates the plan review: presents the plan, addresses objections, revises if needed, and collects votes from the other 8 agents. The Architect does not vote on the plan
- Skills: `brainstorming`, `writing-plans`, `archibald`
- Can: Read code, search, ask questions. Cannot: Write code.

### 2. Tech Lead

- Bidirectional peer relationship with Architect — pushes back on unnecessary design complexity, receives challenges on technical feasibility judgments
- Arbitrates technical disagreements between agents during plan review, except disputes between an agent and the Architect's plan decisions — those escalate to the user since the Tech Lead is a peer to the Architect, not an authority over the Architect's plan
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

- Independently reviews the spec with no knowledge of the Architect's original design notes or any prior reviews of the spec
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
- Responsible for git operations: stages files explicitly (never `git add -A` or `git add .`), writes commit messages that describe what changed and why with no AI attribution, following all git conventions in AGENTS.md
- Skills: `test-driven-development`, `verification-before-completion`
- Can: Read/write code, run tests, git operations. Cannot: Modify specs or deviate from plan.

### 6. Code Simplifier

Runs twice in the pipeline:

- Pre-implementation: Reviews the spec and plan for over-engineering. Can the same behavior be achieved with a simpler design? Flags issues before any code is written.
- Post-implementation: Reviews the actual code for bloat, unnecessary abstractions, dead paths. Makes simplification changes to source code only — cannot modify test files. After making changes, the Tech Lead reviews the Code Simplifier's changes for intent alignment before the pipeline continues.

- Skills: `kiss`, `simplify`
- Can: Read specs/plans, read/modify source code. Cannot: Add features, change behavior, or modify test files.

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
8 agents review plan, Architect presents and defends (consensus required)
  |  |-- Technical disputes between agents -> Tech Lead arbitrates
  |  |-- Disputes with Architect's plan -> escalate to User
  |  '-- Unresolved disputes -> User
  v
User approves plan
  |
  v
Implementer (e2e test from spec first, then implement; Tech Lead available)
  |
  v
Code Simplifier (simplifies source code only, no test files -- POST-IMPL)
  |
  v
Tech Lead (reviews Code Simplifier's changes for intent alignment)
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

## Plan Review Protocol

The plan review is the consensus step where all agents evaluate the implementation plan before execution.

**Participants:** All 9 agents. The Architect presents and defends the plan. The other 8 agents review and vote.

**Review format:** Each reviewing agent produces a structured verdict:
- APPROVE — no issues found from this agent's perspective
- BLOCK (with specific issues) — cannot proceed until these issues are resolved

**Consensus definition:** All 8 reviewing agents must APPROVE. Any BLOCK halts the process.

**When an agent blocks:**
1. The blocking agent states the specific issue and why it blocks from their perspective.
2. The Architect addresses the objection — either by explaining why the plan is correct or by revising the plan.
3. If revised, only the agents that blocked re-review the revised plan. Agents that already approved do not re-review unless the revision materially changed an area they reviewed.
4. If the Architect disagrees with a block:
   - Technical dispute between two non-Architect agents: Tech Lead arbitrates.
   - Dispute between an agent and the Architect's plan: escalates to the user.
   - Tech Lead and Architect disagree: escalates to the user.

**Termination:** If consensus is not reached after 3 revision cycles, the Architect halts the process and escalates the unresolved disagreements to the user with a summary of each position.

## Failure Escalation

Failures in the post-implementation chain don't all route the same way:

| Issue type | Routes to | Example |
|---|---|---|
| Code/test issue | Implementer | Failing test, lint violation, missing assertion |
| Bloat/complexity (in Implementer's code) | Implementer (guided by Code Simplifier's findings) | Unnecessary abstraction, dead code path |
| Bug introduced by Code Simplifier's changes | Code Simplifier | Code Simplifier owns fixes to its own modifications |
| Design-level issue | Architect + Tech Lead redesign | SOLID violation that requires architectural change |
| Intent misalignment | User | Feature behaves correctly but isn't what was asked for |

After any fix, the pipeline re-runs from the point of failure — not from the beginning.

**Loop termination:** If an agent cannot resolve its issue after 3 attempts, it halts and escalates: code/test issues escalate to the Tech Lead for guidance, design issues escalate to the user, intent issues escalate to the user. The pipeline does not loop indefinitely.

## Governance

- Consensus required for plan approval — all 8 reviewing agents must approve (Architect presents, does not vote).
- Tech Lead arbitrates technical disagreements between agents, except disputes involving the Architect's plan decisions which escalate to the user.
- Unresolved disputes escalate to the user — only when Tech Lead cannot resolve or when the dispute involves the Architect.
- The user approves the plan before implementation begins — this is the single mandatory approval gate.
- No agent reviews its own output — enforced by pipeline structure. The Architect's participation in plan review is as presenter/defender, not reviewer.

## Skills Reference

Skills are capabilities defined in the project's superpowers plugin catalog. Each skill provides structured guidance for a specific activity:

| Skill | Purpose | Used by |
|---|---|---|
| `brainstorming` | Collaborative design exploration, approach comparison | Architect |
| `writing-plans` | Structured implementation plan creation | Architect |
| `archibald` | Architecture quality assessment (dependencies, coupling, complexity) | Architect, Tech Lead |
| `solid-principles` | SOLID principle violation detection | Tech Lead, Quality Reviewer |
| `beyond-solid-principles` | Extended architecture principles (DRY, KISS, LoD, resilience) | Tech Lead, Quality Reviewer |
| `spec-writer` | Formal specification document creation | Spec Writer |
| `spec-dd` | Specification-driven development and test derivation | Spec Writer, Spec Reviewer, Test Gap Analyzer |
| `test-driven-development` | TDD workflow: test first, then implement | Implementer |
| `verification-before-completion` | Pre-completion verification checklist | Implementer |
| `kiss` | KISS complexity assessment and simplification opportunities | Code Simplifier |
| `simplify` | Code simplification and bloat removal | Code Simplifier |
| `review-changes` | Structured code review of changes | Quality Reviewer |
