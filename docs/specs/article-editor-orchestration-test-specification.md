# Article Editor Orchestration — Test Specification

## Coverage Matrix

| Spec Requirement | Test Scenario(s) |
|-----------------|------------------|
| AC-1.1: Returns ArticleGenerationResult | TS-1 |
| AC-1.2: SUCCESS when zero concerns | TS-2 |
| AC-1.3: SUCCESS when no misleading verdicts | TS-3 |
| AC-1.4: Compiles feedback and requests revision | TS-4, TS-5 |
| AC-1.5: FAILED after max rounds | TS-6 |
| AC-2.1: Loop step order | TS-7 |
| AC-2.2: Loop runs at most editor_max_rounds | TS-6, TS-8 |
| AC-2.3: Sequential specialist dispatch | TS-9 |
| AC-2.4: Specialist routing by selected_agent | TS-10, TS-11 |
| AC-3.1: No LLM calls in feedback compilation | TS-12 |
| AC-3.2: Verdicts sorted by concern_id | TS-13 |
| AC-3.3: todo_list from REWRITE/REMOVE verdicts | TS-14, TS-15 |
| AC-3.4: improvement_suggestions from KEEP verdicts, max 5 | TS-16 |
| AC-3.5: Rating formula | TS-17, TS-18 |
| AC-3.6: passed always False | TS-14 |
| AC-3.7: Reasoning format | TS-19 |
| AC-3.8: Verdicts included in feedback | TS-14 |
| AC-4.1: Run artifacts directory created | TS-20 |
| AC-4.5: Sequential artifact numbering | TS-21 |
| AC-5.1: Canonical output written | TS-22 |
| AC-5.2: Canonical output shape | TS-23 |
| AC-5.3: Failure still writes canonical output | TS-24 |
| AC-6.1: Missing channel_name/slug/source_file | TS-25, TS-26, TS-27 |
| AC-6.3: publish_date optional | TS-28 |
| AC-7.1: Protocol-based DI | TS-29 |
| AC-7.3: Agent factory mock/default | TS-30, TS-31 |
| AC-8.1: Agent errors propagate | TS-32 |
| AC-8.3: Bullet parser fails on non-bullet text | TS-33 |
| EC-1: Zero concerns first review | TS-2 |
| EC-2: Concerns but none misleading | TS-3 |
| EC-4: Mapping references unknown concern_id | TS-34 |
| EC-5: REWRITE verdict with suggested_fix=None | TS-15 |
| EC-6: editor_max_rounds = 1 | TS-8 |
| EC-7: Revision produces new concerns | TS-5 |
| EC-9: Missing metadata | TS-25, TS-26, TS-27 |

## Test Scenarios

### Happy Path

**TS-1: Orchestrator returns ArticleGenerationResult for valid input**
```
Given a valid source_text, source_metadata (with channel_name, slug, source_file,
      video_id, article_title), style_mode "SCIAM_MAGAZINE", and reader_preference ""
And a mock writer agent that returns a static article draft
And a mock review agent that returns empty bullets
When generate_article() is called
Then the result is an ArticleGenerationResult
And result.success is True
And result.article is not None
And result.metadata is not None
And result.editor_report is not None
And result.artifacts_dir is not None
And result.error is None
```

**TS-2: SUCCESS when article review finds zero concerns**
```
Given a mock writer agent that returns a valid draft
And a mock review agent that returns empty markdown_bullets
When generate_article() is called
Then result.success is True
And result.editor_report.final_status is "SUCCESS"
And result.editor_report.total_iterations is 1
And result.editor_report.iterations is an empty list
And result.editor_report.blocking_concerns is None
```

**TS-3: SUCCESS when specialists find no misleading concerns**
```
Given a mock writer agent that returns a valid draft
And a mock review agent that returns 2 bullet concerns
And a mock concern mapping agent that maps both to "opinion"
And a mock opinion agent that returns misleading=False, status=KEEP for both
When generate_article() is called
Then result.success is True
And result.editor_report.final_status is "SUCCESS"
And result.editor_report.total_iterations is 1
And the iteration report contains 2 concerns, 2 mappings, 2 verdicts
And feedback_to_writer is None (not compiled because passed)
```

### Feedback Loop

**TS-4: Feedback compiled and revision requested when misleading concerns found**
```
Given editor_max_rounds is 3
And a mock writer agent that tracks call count
And a mock review agent that returns 1 concern on the first call
    and returns empty bullets on the second call
And a mock concern mapping agent that maps the concern to "opinion"
And a mock opinion agent that returns misleading=True, status=REWRITE,
    suggested_fix="Fix this"
When generate_article() is called
Then the writer agent was called twice (generate + revise)
And result.success is True
And result.editor_report.total_iterations is 2
And the first iteration report has feedback_to_writer with passed=False
And the second iteration has zero concerns
```

**TS-5: Multi-round revision loop converges to success**
```
Given editor_max_rounds is 3
And a mock review agent that returns concerns on iterations 1 and 2,
    and empty bullets on iteration 3
And mock specialists that return misleading=True on iterations 1 and 2
And a mock writer agent that accepts revisions
When generate_article() is called
Then result.success is True
And result.editor_report.total_iterations is 3
And the writer was called 3 times (1 generate + 2 revise)
```

**TS-6: FAILED after editor_max_rounds with unresolved concerns**
```
Given editor_max_rounds is 2
And a mock review agent that always returns 1 concern
And a mock concern mapping agent that maps to "opinion"
And a mock opinion agent that always returns misleading=True, status=REWRITE
When generate_article() is called
Then result.success is False
And result.error contains "Unresolved misleading concerns after editor_max_rounds"
And result.editor_report.final_status is "FAILED"
And result.editor_report.total_iterations is 2
And result.editor_report.blocking_concerns has 1 concern
And result.article is not None (best available draft)
```

### Loop Mechanics

**TS-7: Loop executes steps in correct order**
```
Given a mock writer that records call order
And a mock review agent that returns 1 concern (then empty on 2nd call)
And a mock concern mapping agent that records call order
And a mock specialist that records call order and returns misleading=True, REWRITE
And editor_max_rounds is 2
When generate_article() is called
Then the call order is:
  1. writer.generate()
  2. review.review()
  3. concern_mapping.map_concerns()
  4. specialist.evaluate()
  5. writer.revise()
  6. review.review() (second iteration, returns empty)
```

**TS-8: editor_max_rounds=1 fails immediately when misleading found**
```
Given editor_max_rounds is 1
And a mock review agent that returns 1 concern
And a mock specialist that returns misleading=True
When generate_article() is called
Then result.success is False
And result.editor_report.total_iterations is 1
And the writer was called exactly once (generate, no revise)
```

**TS-9: Specialists dispatched sequentially in mapping order**
```
Given a mock review agent that returns 3 concerns
And a mock concern mapping agent that maps concern 1 to "fact_check",
    concern 2 to "opinion", concern 3 to "style_review"
And mock specialists that record call order
When generate_article() is called
Then fact_check was called first
And opinion was called second
And style_review was called third
```

### Specialist Routing

**TS-10: Each selected_agent routes to correct specialist**
```
Given a mock review agent that returns 5 concerns
And a mock concern mapping agent that maps each concern to a different specialist:
    concern 1 → "fact_check"
    concern 2 → "evidence_finding"
    concern 3 → "opinion"
    concern 4 → "attribution"
    concern 5 → "style_review"
And each mock specialist records its invocations
When generate_article() is called
Then fact_check agent received concern 1
And evidence_finding agent received concern 2
And opinion agent received concern 3
And attribution agent received concern 4
And style_review agent received concern 5
```

**TS-11: Unknown selected_agent raises ValueError**
```
Given a mock review agent that returns 1 concern
And a mock concern mapping agent that maps it to "unknown_agent"
When generate_article() is called
Then ValueError is raised with message containing "Unknown specialist agent"
```

### Feedback Compilation

**TS-12: Feedback compilation makes no LLM calls**
```
Given verdicts with mixed statuses (KEEP, REWRITE, REMOVE)
When _compile_feedback() is called
Then the result is a WriterFeedback object
And no LLM client calls were made (verify via mock)
```

**TS-13: Verdicts sorted by concern_id in feedback**
```
Given verdicts with concern_ids [3, 1, 2] (unsorted)
When _compile_feedback() is called
Then feedback.verdicts has concern_ids in order [1, 2, 3]
```

**TS-14: todo_list populated from REWRITE/REMOVE verdicts**
```
Given verdicts:
  - concern_id=1, status=REWRITE, suggested_fix="Fix A", misleading=True
  - concern_id=2, status=KEEP, misleading=False
  - concern_id=3, status=REMOVE, suggested_fix="Remove C", misleading=True
When _compile_feedback() is called
Then feedback.todo_list is ["Fix A", "Remove C"]
And feedback.passed is False
And feedback.verdicts contains all 3 verdicts
```

**TS-15: REWRITE verdict with suggested_fix=None excluded from todo_list**
```
Given verdicts:
  - concern_id=1, status=REWRITE, suggested_fix=None, misleading=True
  - concern_id=2, status=REWRITE, suggested_fix="Fix B", misleading=True
When _compile_feedback() is called
Then feedback.todo_list is ["Fix B"] (concern 1 excluded)
```

**TS-16: improvement_suggestions limited to 5 items**
```
Given 7 verdicts all with status=KEEP and rationale "Reason N"
When _compile_feedback() is called
Then feedback.improvement_suggestions has exactly 5 items
And they are the rationales from the first 5 by concern_id order
```

**TS-17: Rating formula basic calculation**
```
Given 2 REWRITE verdicts with suggested_fix and 1 KEEP verdict
When _compile_feedback() is called
Then rating = max(1, min(10, 10 - 2*2 - 1*1)) = 5
```

**TS-18: Rating clamped to minimum 1**
```
Given 5 REWRITE verdicts with suggested_fix and 3 KEEP verdicts
When _compile_feedback() is called
Then rating = max(1, min(10, 10 - 2*5 - 1*3)) = max(1, -3) = 1
```

**TS-19: Reasoning format with empty vs non-empty todos**
```
Given no REWRITE or REMOVE verdicts (only KEEP)
When _compile_feedback() is called
Then feedback.reasoning is "No required rewrites or removals were identified."

Given 2 REWRITE verdicts with suggested_fix "Fix A" and "Fix B"
When _compile_feedback() is called
Then feedback.reasoning is "Required changes:\n1. Fix A\n2. Fix B"
```

### Artifact Logging

**TS-20: Run artifacts directory created with correct structure**
```
Given channel_name "TestChannel", slug "test-article",
      source_file "transcript.txt", style_mode "SCIAM_MAGAZINE"
When generate_article() is called
Then a directory exists at {run_artifacts_dir}/TestChannel/test-article/{run_id}/
And the run_id matches pattern YYYYMMDDTHHMMSSZ_{8_hex_chars}
```

**TS-21: Artifact files numbered sequentially**
```
Given a full run with 1 iteration that finds concerns and revises
When generate_article() completes
Then artifact files exist with 3-digit prefix numbers
And prefixes increase monotonically
And final artifacts have prefix >= 900
```

### Canonical Output

**TS-22: Canonical output file written on success**
```
Given a successful run (review finds no concerns)
When generate_article() completes
Then a file exists at {final_articles_dir}/{channel_name}/{slug}.json
And the file contains valid JSON
```

**TS-23: Canonical output contains complete ArticleGenerationResult**
```
Given a successful run
When the canonical output file is read
Then it contains keys: success, article, metadata, editor_report, artifacts_dir, error
And article contains: headline, alternative_headline, article_body, description
And metadata contains: source_file, channel_name, video_id, article_title, slug,
    publish_date, references, style_mode, generated_at
And generated_at is an ISO 8601 UTC timestamp
```

**TS-24: Canonical output written on failure**
```
Given a failed run (misleading concerns after max rounds)
When generate_article() completes
Then a canonical output file exists
And its content has success=False
And error is non-null
And article contains the best available draft (not null)
And editor_report.final_status is "FAILED"
```

### Metadata Validation

**TS-25: Missing channel_name raises ValueError**
```
Given source_metadata with channel_name=None
When generate_article() is called
Then ValueError is raised with message "Missing required metadata field: channel_name"
And no agent calls were made
```

**TS-26: Missing slug raises ValueError**
```
Given source_metadata with slug=None
When generate_article() is called
Then ValueError is raised with message "Missing required metadata field: slug"
```

**TS-27: Missing source_file raises ValueError**
```
Given source_metadata with source_file=None
When generate_article() is called
Then ValueError is raised with message "Missing required metadata field: source_file"
```

**TS-28: Null publish_date accepted without error**
```
Given source_metadata with publish_date=None and all other required fields present
When generate_article() completes successfully
Then result.metadata.publish_date is None
And no error was raised
```

### Dependency Injection

**TS-29: Orchestrator accepts mock agents via protocol typing**
```
Given mock implementations of all 8 agent protocols
When ChiefEditorOrchestrator is constructed with mock agents
Then construction succeeds without type errors
And generate_article() can be called successfully
```

**TS-30: Agent factory creates mock agents when agent_name is "mock"**
```
Given a config where all agent_name values are "mock"
When build_chief_editor_orchestrator() is called
Then the returned orchestrator uses mock agent implementations
And generate_article() succeeds without any LLM calls
```

**TS-31: Agent factory raises on invalid agent_name**
```
Given a config where writer agent_name is "invalid_name"
When build_chief_editor_orchestrator() is called
Then ValueError is raised
```

### Error Handling

**TS-32: Agent exception propagates to caller**
```
Given a mock writer agent that raises RuntimeError("LLM failed")
When generate_article() is called
Then RuntimeError is raised with message "LLM failed"
```

**TS-33: Bullet parser raises on non-bullet text**
```
Given a mock review agent that returns "This is not a bullet list"
When generate_article() is called
Then ValueError is raised from the bullet parser
```

**TS-34: Mapping references non-existent concern_id**
```
Given a mock review agent that returns 1 concern with concern_id=1
And a mock concern mapping agent that returns a mapping for concern_id=99
When generate_article() is called
Then ValueError is raised with message "Concern id from mapping not found: 99"
```

## Traceability

Every acceptance criterion (AC-1.1 through AC-8.3) and edge case (EC-1 through EC-9) from the behavioral specification is covered by at least one test scenario. The coverage matrix at the top of this document provides the complete mapping.
