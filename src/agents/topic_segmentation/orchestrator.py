"""Orchestrator for topic segmentation agent workflow."""

from collections.abc import Callable

from src.agents.topic_segmentation.agent import TopicSegmentationAgent
from src.agents.topic_segmentation.critic import TopicSegmentationCritic
from src.agents.topic_segmentation.models import AgentSegmentationResponse, SegmentationAttempt, SegmentationResult
from src.config import Config, TopicSegmentationConfig


class TopicSegmentationOrchestrator:
    """Orchestrates the multi-agent topic segmentation workflow."""

    def __init__(self, ts_config: TopicSegmentationConfig, config: Config) -> None:
        """Initialize the orchestrator.

        Args:
            ts_config: Topic segmentation configuration.
            config: Full application configuration.

        Raises:
            KeyError: If required environment variables are missing.
        """
        self._config = ts_config

        # Initialize agents
        self._agent = TopicSegmentationAgent(llm_config=ts_config.agent_llm, config=config)

        self._critic = TopicSegmentationCritic(llm_config=ts_config.critic_llm, config=config)

    def segment_transcript(
        self,
        simplified_transcript: str,
        on_agent_response: Callable[[int, AgentSegmentationResponse], None],
    ) -> SegmentationResult:
        """Segment a transcript with agent-critic feedback loop.

        Args:
            simplified_transcript: Simplified format transcript.
            on_agent_response: Callback invoked after each successful agent response.
                Takes attempt number and agent response as arguments.

        Returns:
            Complete segmentation result with all attempts tracked.
        """
        attempts: list[SegmentationAttempt] = []
        retry_limit = self._config.retry_limit

        for attempt_num in range(1, retry_limit + 1):
            print(f"      [Orchestrator] Starting attempt {attempt_num}/{retry_limit}")

            # Determine if this is a retry
            retry_feedback = None
            if attempt_num > 1:
                # Get feedback from previous attempt
                previous_attempt = attempts[-1]
                if previous_attempt.critic_rating is None:
                    # Should never happen, but handle gracefully
                    break

                retry_feedback = {
                    "rating": previous_attempt.critic_rating.rating,
                    "pass": previous_attempt.critic_rating.pass_,
                    "reasoning": previous_attempt.critic_rating.reasoning,
                    "improvement_suggestions": previous_attempt.critic_rating.improvement_suggestions,
                }

            # Attempt segmentation
            try:
                agent_response = self._agent.segment(
                    simplified_transcript=simplified_transcript,
                    retry_feedback=retry_feedback,
                )
            except ValueError as e:
                # Agent produced invalid response
                return SegmentationResult(
                    success=False,
                    attempts=attempts,
                    best_attempt=self._select_best_attempt(attempts),
                    failure_reason=f"Agent error on attempt {attempt_num}: {e}",
                )

            # Invoke callback immediately after successful agent response
            on_agent_response(attempt_num, agent_response)

            # Evaluate with critic
            try:
                critic_rating = self._critic.evaluate(
                    simplified_transcript=simplified_transcript,
                    segmentation=agent_response,
                )
            except ValueError as e:
                # Critic produced invalid response
                return SegmentationResult(
                    success=False,
                    attempts=attempts,
                    best_attempt=self._select_best_attempt(attempts),
                    failure_reason=f"Critic error on attempt {attempt_num}: {e}",
                )

            # Log critic evaluation
            print(f"      [Critic] Rating: {critic_rating.rating}, Pass: {critic_rating.pass_}")
            print(f"      [Critic] Reasoning: {critic_rating.reasoning}")
            if not critic_rating.pass_:
                print(f"      [Critic] Suggestions: {critic_rating.improvement_suggestions}")

            # Record this attempt
            attempt = SegmentationAttempt(
                attempt_number=attempt_num,
                response=agent_response,
                critic_rating=critic_rating,
            )
            attempts.append(attempt)

            # Check if we should accept this result
            if critic_rating.pass_:
                # Success!
                return SegmentationResult(
                    success=True,
                    attempts=attempts,
                    best_attempt=attempt,
                    failure_reason=None,
                )

            # If this was the last attempt, we're done
            if attempt_num == retry_limit:
                break

        # Exhausted retries without success
        return SegmentationResult(
            success=False,
            attempts=attempts,
            best_attempt=self._select_best_attempt(attempts),
            failure_reason=f"Exhausted retry limit ({retry_limit}) without passing",
        )

    def _select_best_attempt(self, attempts: list[SegmentationAttempt]) -> SegmentationAttempt | None:
        """Select the best attempt from a list.

        Ranking logic:
        1. "great" rating > "ok" rating > "bad" rating
        2. If tied, prefer later attempt (refinement)

        Args:
            attempts: List of segmentation attempts.

        Returns:
            Best attempt, or None if list is empty.
        """
        if not attempts:
            return None

        # Define rating priority
        rating_priority = {"great": 3, "ok": 2, "bad": 1}

        # Sort attempts by rating (descending) then attempt number (descending)
        sorted_attempts = sorted(
            attempts,
            key=lambda a: (
                rating_priority[a.critic_rating.rating if a.critic_rating else "bad"],
                a.attempt_number,
            ),
            reverse=True,
        )

        return sorted_attempts[0]
