"""Orchestrator for topic segmentation agent workflow."""

from src.agents.topic_segmentation.agent import TopicSegmentationAgent
from src.agents.topic_segmentation.critic import TopicSegmentationCritic
from src.agents.topic_segmentation.models import SegmentationAttempt, SegmentationResult
from src.config import TopicSegmentationConfig


class TopicSegmentationOrchestrator:
    """Orchestrates the multi-agent topic segmentation workflow."""

    def __init__(self, config: TopicSegmentationConfig) -> None:
        """Initialize the orchestrator.

        Args:
            config: Topic segmentation configuration.

        Raises:
            KeyError: If required environment variables are missing.
        """
        self._config = config

        # Initialize agents
        self._agent = TopicSegmentationAgent(llm_config=config.agent_llm)

        self._critic = TopicSegmentationCritic(llm_config=config.critic_llm)

    def segment_transcript(
        self,
        video_id: str,
        video_title: str,
        channel_name: str,
        simplified_transcript: str,
    ) -> SegmentationResult:
        """Segment a transcript with agent-critic feedback loop.

        Args:
            video_id: Unique video identifier.
            video_title: Video title.
            channel_name: YouTube channel name.
            simplified_transcript: Simplified format transcript.

        Returns:
            Complete segmentation result with all attempts tracked.
        """
        attempts: list[SegmentationAttempt] = []
        retry_limit = self._config.retry_limit

        for attempt_num in range(1, retry_limit + 1):
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
                    video_id=video_id,
                    video_title=video_title,
                    channel_name=channel_name,
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
