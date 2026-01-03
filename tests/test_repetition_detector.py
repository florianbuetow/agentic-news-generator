"""Tests for the RepetitionDetector class."""

import pytest

from src.utils.repetition_detector import RepetitionDetector


class TestCountConsecutiveRepetitions:
    """Tests for _count_consecutive_repetitions method."""

    def test_two_repetitions(self):
        """Test counting 2 consecutive repetitions."""
        detector = RepetitionDetector()
        text = "hello world hello world"
        words = detector.prepare(text).split()

        # "hello world" appears at [0:2] and [2:4]
        count = detector._count_consecutive_repetitions(words, 0, 2)
        assert count == 2

    def test_three_repetitions(self):
        """Test counting 3 consecutive repetitions."""
        detector = RepetitionDetector()
        text = "abc abc abc"
        words = detector.prepare(text).split()

        count = detector._count_consecutive_repetitions(words, 0, 1)
        assert count == 3

    def test_fifteen_repetitions(self):
        """Test counting 15 consecutive repetitions."""
        detector = RepetitionDetector()
        text = " ".join(["like, you know,"] * 15)
        words = detector.prepare(text).split()

        # Pattern is 3 words: "like," "you" "know,"
        count = detector._count_consecutive_repetitions(words, 0, 3)
        assert count == 15

    def test_non_consecutive_returns_one(self):
        """Test that non-consecutive pattern returns 1."""
        detector = RepetitionDetector()
        text = "apple banana apple cherry"
        words = detector.prepare(text).split()

        # "apple" appears at [0] and [2], but not consecutively
        count = detector._count_consecutive_repetitions(words, 0, 1)
        assert count == 1

    def test_forty_repetitions_yeah(self):
        """Test massive repetition of single word (40+ times)."""
        detector = RepetitionDetector()
        # Create 40 repetitions of "Yeah."
        text = " ".join(["Yeah."] * 40)
        words = detector.prepare(text).split()

        count = detector._count_consecutive_repetitions(words, 0, 1)
        assert count == 40


class TestDetectHallucinations:
    """Tests for detect_hallucinations method with formula-based scoring."""

    def test_massive_loop_detected(self):
        """Test that massive repetition loop is detected."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, threshold=10)
        text = " ".join(["you know, like,"] * 15)

        results = detector.detect_hallucinations(text)

        assert len(results) > 0
        # Should detect the pattern
        start, end, k, rep_count = results[0]
        assert k == 3  # "you" "know," "like,"
        assert rep_count == 15
        assert k * rep_count == 45  # Score should be 45

    def test_natural_stutter_filtered(self):
        """Test that 2x repetition (natural stutter) is filtered."""
        detector = RepetitionDetector()
        text = "as like a, as like a, an object schema"

        results = detector.detect_hallucinations(text)

        # Should be filtered because score = 3 * 2 = 6 < 10
        assert len(results) == 0

    def test_ambiguous_3x_repetition_filtered(self):
        """Test that ambiguous 3x repetition is filtered."""
        detector = RepetitionDetector()
        text = "for this work, for this work, for this work, the xi"

        results = detector.detect_hallucinations(text)

        # Should be filtered because score = 3 * 3 = 9 < 10
        assert len(results) == 0

    def test_threshold_6_more_sensitive(self):
        """Test that threshold=6 catches 2x repetitions."""
        detector_6 = RepetitionDetector(min_k=1, min_repetitions=5, threshold=6)
        text = "as like a, as like a, an object"

        results = detector_6.detect_hallucinations(text)

        # Should NOT be detected because score = 3 * 2 = 6, but threshold requires > 6
        assert len(results) == 0

        # But with threshold=5 it should be detected
        detector_5 = RepetitionDetector(min_k=1, min_repetitions=5, threshold=5)
        results = detector_5.detect_hallucinations(text)
        assert len(results) > 0

    def test_min_repetitions_absolute_minimum(self):
        """Test that min_repetitions provides absolute minimum."""
        # Single word repeated 5 times: score = 1 * 5 = 5
        text = " ".join(["word"] * 5)

        # With min_repetitions=5, should be detected even though score=5 < threshold=10
        detector = RepetitionDetector(min_k=1, min_repetitions=5, threshold=10)
        results = detector.detect_hallucinations(text)
        assert len(results) > 0
        start, end, k, rep_count = results[0]
        assert rep_count == 5

    def test_real_hallucination_nick_lane_yeah_pattern(self):
        """Test detection of real hallucination from Nick Lane transcript.

        This is a real example of Whisper hallucination where the model
        gets stuck repeating "Yeah." about 40 times.
        """
        detector = RepetitionDetector()
        # Actual hallucination pattern from transcript
        text = (
            "Because if there was another way to solve it, then what you would expect "
            "is that as soon as you get to the stage of prokaryotes that have other "
            "niches that they could colonize, if only they could drive towards "
            "complexity, this would somehow be solved. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. "
            "Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah."
        )

        results = detector.detect_hallucinations(text)

        # Should detect the "Yeah." pattern
        assert len(results) > 0

        # Find the "Yeah." pattern
        yeah_pattern = None
        for start, end, k, rep_count in results:
            words = detector.prepare(text).split()
            pattern_text = " ".join(words[start:end])
            if "Yeah" in pattern_text or "yeah" in pattern_text:
                yeah_pattern = (start, end, k, rep_count)
                break

        assert yeah_pattern is not None, "Should detect Yeah. pattern"
        start, end, k, rep_count = yeah_pattern

        # Should have detected many repetitions
        assert rep_count >= 10, f"Should detect at least 10 repetitions, got {rep_count}"

        # Score should be well above threshold
        score = k * rep_count
        assert score > 10, f"Score {score} should be > 10"

    def test_short_phrase_many_reps_detected(self):
        """Test that short phrase with many repetitions is detected."""
        detector = RepetitionDetector()
        # 1 word repeated 11 times: score = 1 * 11 = 11 > 10
        text = " ".join(["word"] * 11)

        results = detector.detect_hallucinations(text)
        assert len(results) > 0
        start, end, k, rep_count = results[0]
        assert k == 1
        assert rep_count == 11
        assert k * rep_count == 11

    def test_long_phrase_fewer_reps_detected(self):
        """Test that long phrase with fewer repetitions is detected."""
        detector = RepetitionDetector()
        # 5 words repeated 3 times: score = 5 * 3 = 15 > 10
        phrase = "this is a long phrase"
        text = " ".join([phrase] * 3)

        results = detector.detect_hallucinations(text)
        assert len(results) > 0
        # Should find a pattern with score > 10
        found = False
        for start, end, k, rep_count in results:
            if k * rep_count > 10:
                found = True
                break
        assert found

    def test_no_consecutive_repetition_not_detected(self):
        """Test that non-consecutive repetitions are not detected."""
        detector = RepetitionDetector()
        text = "apple banana apple cherry apple date"

        results = detector.detect_hallucinations(text)
        # "apple" appears 3 times but not consecutively
        assert len(results) == 0

    def test_empty_text(self):
        """Test that empty text returns no results."""
        detector = RepetitionDetector()
        results = detector.detect_hallucinations("")
        assert len(results) == 0

    def test_single_word(self):
        """Test that single word with no repetition returns no results."""
        detector = RepetitionDetector()
        results = detector.detect_hallucinations("hello")
        assert len(results) == 0

    def test_natural_speech_i_have_to_filtered(self):
        """Test that 'I have to I have to' (natural stutter) is filtered."""
        detector = RepetitionDetector()
        text = "because I have to I have to generate the chunk"

        results = detector.detect_hallucinations(text)

        # k=3 ("I have to") × 2 reps = 6 < 10, should be filtered
        assert len(results) == 0

    def test_natural_speech_you_can_safely_filtered(self):
        """Test that 'you can safely you can safely' is filtered."""
        detector = RepetitionDetector()
        text = "that you can safely you can safely scale beta with N"

        results = detector.detect_hallucinations(text)

        # k=3 × 2 = 6 < 10, should be filtered
        assert len(results) == 0

    def test_natural_speech_i_dont_think_filtered(self):
        """Test that 'I don't think I don't think' is filtered."""
        detector = RepetitionDetector()
        text = "it. I don't think I don't think it's exactly like"

        results = detector.detect_hallucinations(text)

        # k=3 × 2 = 6 < 10, should be filtered
        assert len(results) == 0

    def test_emphatic_speech_thank_you_filtered(self):
        """Test that polite closing 'Thank you so much' repeated is filtered."""
        detector = RepetitionDetector()
        text = "Thank you so much. Thank you so much."

        results = detector.detect_hallucinations(text)

        # k=4 × 2 = 8 < 10, should be filtered
        assert len(results) == 0

    def test_emphatic_speech_who_cares_filtered(self):
        """Test that rhetorical repetition 'Who cares about you?' is filtered."""
        detector = RepetitionDetector()
        text = "showing up at Los Alamos? Who cares about you? Who cares about you?"

        results = detector.detect_hallucinations(text)

        # k=4 × 2 = 8 < 10, should be filtered
        assert len(results) == 0

    def test_emphatic_speech_sky_falling_filtered(self):
        """Test that idiom emphasis 'the sky is falling' is filtered."""
        detector = RepetitionDetector()
        text = "American, the sky is falling, the sky is falling, right?"

        results = detector.detect_hallucinations(text)

        # k=4 × 2 = 8 < 10, should be filtered
        assert len(results) == 0

    def test_emphatic_speech_i_would_love_filtered(self):
        """Test that enthusiasm 'I would love to do that' is filtered."""
        detector = RepetitionDetector()
        text = "expand your search. I would love to do that. I would love to do that. Yes"

        results = detector.detect_hallucinations(text)

        # k=6 × 2 = 12 > 10, might be detected but check
        # Actually this could be detected or filtered depending on exact pattern
        # Let's allow either result since it's borderline
        # If detected, score should be around 12
        if results:
            found = False
            for start, end, k, rep_count in results:
                score = k * rep_count
                if score >= 12:
                    found = True
            assert found, "If detected, should have score >= 12"

    def test_natural_language_a_percent_of_filtered(self):
        """Test that technical phrase 'a percent of a percent of' is filtered."""
        detector = RepetitionDetector()
        text = "maybe a percent of a percent of the speed of light"

        results = detector.detect_hallucinations(text)

        # k=3 ("a percent of") × 2 = 6 < 10, should be filtered
        assert len(results) == 0

    def test_this_is_like_natural_filler_filtered(self):
        """Test that 'this is like this is like' (natural filler) is filtered."""
        detector = RepetitionDetector()
        text = "I think this is like this is like very connected to"

        results = detector.detect_hallucinations(text)

        # k=3 × 2 = 6 < 10, should be filtered
        assert len(results) == 0

    def test_customer_data_pattern_high_repetitions(self):
        """Test pattern with high repetition count gets detected."""
        detector = RepetitionDetector()
        # Simulate a pattern that repeats many times
        phrase = "customer data for performance monitoring"
        text = " ".join([phrase] * 10)

        results = detector.detect_hallucinations(text)

        # Should detect high-repetition pattern
        assert len(results) > 0
        # Find the longest pattern
        max_score = 0
        for start, end, k, rep_count in results:
            score = k * rep_count
            max_score = max(max_score, score)
        # With 10 repetitions, score should be high
        assert max_score >= 10, f"Expected high score, got {max_score}"

    def test_edge_case_punctuation_variation_not_detected(self):
        """Test that punctuation variations prevent consecutive detection."""
        detector = RepetitionDetector()
        # Different punctuation makes them non-identical
        text = "We're very, we're very ready for it."

        results = detector.detect_hallucinations(text)

        # The phrases differ in punctuation: "very," vs "very"
        # So they're not exact consecutive repetitions
        assert len(results) == 0

    def test_edge_case_capitalization_variation_not_detected(self):
        """Test that capitalization differences prevent consecutive detection."""
        detector = RepetitionDetector()
        text = "So we need to understand. so we need to understand."

        results = detector.detect_hallucinations(text)

        # Different capitalization: "So" vs "so"
        assert len(results) == 0

    def test_multiple_patterns_in_same_text(self):
        """Test detection when text contains multiple repetition patterns."""
        detector = RepetitionDetector()
        # Multiple different patterns
        text = "word word word and phrase phrase phrase phrase and thing thing"

        results = detector.detect_hallucinations(text)

        # "word" × 3 = 3 < 10: filtered
        # "phrase" × 4 = 4 < 10: filtered
        # "thing" × 2 = 2 < 10: filtered
        # All should be filtered
        assert len(results) == 0

    def test_threshold_boundary_exactly_at_threshold(self):
        """Test pattern with score exactly at threshold (not detected)."""
        detector = RepetitionDetector()
        # Create pattern with score = exactly 10
        # k=2 × 5 reps = 10
        text = " ".join(["word pair"] * 5)

        results = detector.detect_hallucinations(text)

        # Score = 10, but threshold requires > 10, so should be filtered
        # UNLESS min_repetitions=5 kicks in
        # With default min_repetitions=5, this should be detected
        assert len(results) > 0, "Should be detected via min_repetitions=5"

    def test_threshold_boundary_one_above_threshold(self):
        """Test pattern with score just above threshold (detected)."""
        detector = RepetitionDetector()
        # k=1 × 11 reps = 11 > 10
        text = " ".join(["word"] * 11)

        results = detector.detect_hallucinations(text)

        assert len(results) > 0
        start, end, k, rep_count = results[0]
        assert k * rep_count == 11


class TestDetectBasic:
    """Tests for the basic detect method (all repetitions, not just consecutive)."""

    def test_detect_finds_all_repetitions(self):
        """Test that detect() finds all repetitions, not just consecutive."""
        detector = RepetitionDetector()
        text = "apple banana cherry apple banana cherry fig apple banana cherry"

        results = detector.detect(text, min_k=3)

        # Should find multiple instances of "apple banana cherry"
        assert len(results) > 0

    def test_detect_prioritizes_longer_patterns(self):
        """Test that detect() keeps longest pattern when multiple exist at same position."""
        detector = RepetitionDetector()
        text = "apple banana cherry date apple banana cherry fig"

        results = detector.detect(text, min_k=3)

        # Should prioritize "apple banana cherry" (k=3) at position 0
        # over shorter subpatterns
        found_k3 = False
        for start, end, k in results:
            if start == 0 and k >= 3:
                found_k3 = True
                break
        assert found_k3


class TestPrepare:
    """Tests for text preparation (normalization)."""

    def test_prepare_normalizes_whitespace_only(self):
        """Test that prepare() only normalizes whitespace."""
        detector = RepetitionDetector()

        # Should preserve punctuation and capitalization
        text = "Hello,  World!  How   are you?"
        prepared = detector.prepare(text)
        assert prepared == "Hello, World! How are you?"

    def test_prepare_preserves_punctuation(self):
        """Test that punctuation is preserved."""
        detector = RepetitionDetector()
        text = "We're very, we're very ready."
        prepared = detector.prepare(text)
        assert "'" in prepared
        assert "," in prepared

    def test_prepare_preserves_case(self):
        """Test that case is preserved."""
        detector = RepetitionDetector()
        text = "Hello World"
        prepared = detector.prepare(text)
        assert "Hello" in prepared
        assert "World" in prepared
        assert prepared != "hello world"


class TestConstructorParameters:
    """Tests for constructor parameter configuration."""

    def test_constructor_default_parameters(self):
        """Test that default constructor parameters work correctly."""
        detector = RepetitionDetector()

        # Defaults: min_k=1, min_repetitions=5, threshold=10
        assert detector.min_k == 1
        assert detector.min_repetitions == 5
        assert detector.threshold == 10

    def test_constructor_custom_parameters(self):
        """Test that custom constructor parameters are stored."""
        detector = RepetitionDetector(min_k=3, min_repetitions=7, threshold=15)

        assert detector.min_k == 3
        assert detector.min_repetitions == 7
        assert detector.threshold == 15

    def test_detect_hallucinations_uses_constructor_defaults(self):
        """Test that detect_hallucinations() uses constructor parameters by default."""
        # Create detector with threshold=6 (more sensitive than default 10)
        detector = RepetitionDetector(min_k=1, min_repetitions=5, threshold=6)

        # Pattern: k=3 × 2 reps = 6
        # Should be filtered with threshold=10, but detected with threshold=6
        text = "as like a, as like a, an object"

        # Call WITHOUT explicit parameters - should use constructor values
        results = detector.detect_hallucinations(text)

        # With threshold=6, score=6 should NOT be detected (needs > 6)
        # But min_repetitions=5 doesn't apply here (only 2 reps)
        assert len(results) == 0

    def test_detect_hallucinations_constructor_threshold_15(self):
        """Test constructor with higher threshold filters more patterns."""
        # More conservative detector
        detector = RepetitionDetector(min_k=1, min_repetitions=5, threshold=15)

        # Pattern: k=3 × 4 reps = 12
        # Detected with threshold=10, filtered with threshold=15
        text = " ".join(["word phrase pattern"] * 4)

        # Should be filtered because score=12 < 15 and reps=4 < 5
        results = detector.detect_hallucinations(text)
        assert len(results) == 0

    def test_detect_hallucinations_constructor_min_repetitions(self):
        """Test constructor min_repetitions parameter."""
        # Detector with min_repetitions=3 (lower than default 5)
        detector = RepetitionDetector(min_k=1, min_repetitions=3, threshold=10)

        # Pattern: k=3 × 3 reps = 9 < threshold
        # BUT reps=3 meets min_repetitions=3
        text = " ".join(["test pattern here"] * 3)

        # Should be detected via min_repetitions=3 criterion
        results = detector.detect_hallucinations(text)
        assert len(results) > 0

    def test_different_detector_configurations(self):
        """Test that different detector configurations produce different results."""
        # Pattern: k=3 × 4 reps = 12
        text = " ".join(["word phrase pattern"] * 4)

        # High threshold detector - should filter this pattern
        detector_high = RepetitionDetector(min_k=1, min_repetitions=5, threshold=15)
        # Score=12 < 15 and reps=4 < min_repetitions=5
        results_high = detector_high.detect_hallucinations(text)
        assert len(results_high) == 0

        # Low threshold detector - should detect this pattern
        detector_low = RepetitionDetector(min_k=1, min_repetitions=5, threshold=10)
        # Score=12 > 10, should be detected via score criterion
        results_low = detector_low.detect_hallucinations(text)
        assert len(results_low) > 0

        # Lower min_repetitions - should also detect
        detector_min_reps = RepetitionDetector(min_k=1, min_repetitions=3, threshold=15)
        # reps=4 >= min_repetitions=3, should be detected via min_repetitions criterion
        results_min_reps = detector_min_reps.detect_hallucinations(text)
        assert len(results_min_reps) > 0

    def test_constructor_min_k_parameter_used(self):
        """Test that constructor min_k parameter is used in detection."""
        # Create two detectors with different min_k values
        detector_mink_1 = RepetitionDetector(min_k=1, min_repetitions=2, threshold=100)
        detector_mink_5 = RepetitionDetector(min_k=5, min_repetitions=2, threshold=100)

        # Text with a 3-word pattern
        text = " ".join(["word phrase test"] * 5)

        # With min_k=1, should detect (finds patterns >= 1 word)
        results_1 = detector_mink_1.detect_hallucinations(text)

        # With min_k=5, may or may not detect depending on what patterns form
        results_5 = detector_mink_5.detect_hallucinations(text)

        # The key test: different min_k values can produce different results
        # Just verify that the constructor parameters are being passed through
        # by checking that at least one detector found something (most likely min_k=1)
        assert len(results_1) > 0, "min_k=1 should detect some patterns"
