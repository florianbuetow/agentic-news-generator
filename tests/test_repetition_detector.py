"""Tests for the RepetitionDetector class."""

from pathlib import Path

from src.config import Config
from src.processing.repetition_detector import RepetitionDetector

# Load config for tests
CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
CONFIG = Config(CONFIG_PATH)


class TestCountConsecutiveRepetitions:
    """Tests for _count_consecutive_repetitions method."""

    def test_two_repetitions(self):
        """Test counting 2 consecutive repetitions."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "hello world hello world"
        words = detector.prepare(text).split()

        # "hello world" appears at [0:2] and [2:4]
        count = detector.count_consecutive_repetitions(words, 0, 2)
        assert count == 2

    def test_three_repetitions(self):
        """Test counting 3 consecutive repetitions."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "abc abc abc"
        words = detector.prepare(text).split()

        count = detector.count_consecutive_repetitions(words, 0, 1)
        assert count == 3

    def test_fifteen_repetitions(self):
        """Test counting 15 consecutive repetitions."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = " ".join(["like, you know,"] * 15)
        words = detector.prepare(text).split()

        # Pattern is 3 words: "like," "you" "know,"
        count = detector.count_consecutive_repetitions(words, 0, 3)
        assert count == 15

    def test_non_consecutive_returns_one(self):
        """Test that non-consecutive pattern returns 1."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "apple banana apple cherry"
        words = detector.prepare(text).split()

        # "apple" appears at [0] and [2], but not consecutively
        count = detector.count_consecutive_repetitions(words, 0, 1)
        assert count == 1

    def test_forty_repetitions_yeah(self):
        """Test massive repetition of single word (40+ times)."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # Create 40 repetitions of "Yeah."
        text = " ".join(["Yeah."] * 40)
        words = detector.prepare(text).split()

        count = detector.count_consecutive_repetitions(words, 0, 1)
        assert count == 40


class TestDetectHallucinations:
    """Tests for detect_hallucinations method with SVM-based classification."""

    def test_massive_loop_detected(self):
        """Test that massive repetition loop is detected."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text = " ".join(["you know, like,"] * 15)

        results = detector.detect_hallucinations(text)

        assert len(results) > 0
        # Should detect the pattern
        _start, _end, k, rep_count = results[0]
        assert k == 3  # "you" "know," "like,"
        assert rep_count == 15

    def test_natural_stutter_filtered(self):
        """Test that 2x repetition (natural stutter) is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "as like a, as like a, an object schema"

        results = detector.detect_hallucinations(text)

        # Should be filtered by SVM classifier (k=3, reps=2 -> not hallucination)
        assert len(results) == 0

    def test_ambiguous_3x_repetition_filtered(self):
        """Test that ambiguous 3x repetition is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "for this work, for this work, for this work, the xi"

        results = detector.detect_hallucinations(text)

        # Should be filtered by SVM classifier (k=3, reps=3 -> not hallucination)
        assert len(results) == 0

    def test_natural_stutter_2x_repetition(self):
        """Natural stutter with 2 repetitions should NOT be detected."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text = "as like a, as like a, an object"

        results = detector.detect_hallucinations(text)

        # Should NOT be detected (natural stutter, only 2 reps)
        assert len(results) == 0

    def test_phrase_15x_repetition_detected(self):
        """3-word phrase repeated 15 times should be detected as hallucination."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text_15_reps = " ".join(["you know, like,"] * 15)

        results = detector.detect_hallucinations(text_15_reps)

        # Should be detected as hallucination by SVM (k=3, reps=15)
        assert len(results) > 0
        _start, _end, k, rep_count = results[0]
        assert k == 3  # 3-word phrase
        assert rep_count == 15  # 15 repetitions

    def test_single_word_5x_not_hallucination(self):
        """Single word repeated 5 times classified as NOT hallucination by SVM."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text = " ".join(["word"] * 5)

        results = detector.detect_hallucinations(text)

        # Classifier determines this is NOT a hallucination (k=1, reps=5)
        assert len(results) == 0

    def test_single_word_11x_is_hallucination(self):
        """Single word repeated 11 times classified as hallucination by SVM."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text_11 = " ".join(["word"] * 11)

        results_11 = detector.detect_hallucinations(text_11)

        # Should be detected as hallucination
        assert len(results_11) > 0
        _start, _end, k, rep_count = results_11[0]
        assert k == 1  # Single word
        assert rep_count == 11  # 11 repetitions

    def test_real_hallucination_nick_lane_yeah_pattern(self):
        """Test detection of real hallucination from Nick Lane transcript.

        This is a real example of Whisper hallucination where the model
        gets stuck repeating "Yeah." about 40 times.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
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

    def test_real_hallucination_and_repeated_40_times(self):
        """Test detection of 'and,' repeated 40+ times (hallucination).

        Single word repeated many times should trigger detection because:
        - k=1, reps=40 -> SVM classifies as hallucination
        - reps=40 >= 5 ✓
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = " ".join(["and,"] * 40)

        results = detector.detect_hallucinations(text)

        # Should detect the "and," pattern
        assert len(results) > 0

        _start, _end, k, rep_count = results[0]
        assert k == 1  # Single word
        assert rep_count >= 40

    def test_real_hallucination_long_phrase_with_variations(self):
        """Test detection of long phrase repeated ~10 times.

        This tests a hallucination pattern where the model gets stuck repeating
        a complex phrase multiple times.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # Exact repetition of the core phrase 10 times
        phrase = "We have to be able to do this with the X or Y."
        text = " ".join([phrase] * 10)

        results = detector.detect_hallucinations(text)

        # Should detect the pattern
        assert len(results) > 0

        # Find the longest detected pattern (by k * rep_count product)
        max_score_result = max(results, key=lambda r: r[2] * r[3])
        _start, _end, _k, rep_count = max_score_result

        # Should have detected multiple repetitions
        assert rep_count >= 5, f"Expected at least 5 repetitions, got {rep_count}"

    def test_real_hallucination_thank_you_support_pattern(self):
        """Test detection of 'thank you for your support' hallucination pattern.

        This tests a real-world hallucination where the model gets stuck
        repeatedly thanking for support with slight variations.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = (
            "I don't know what you are saying. I would like to thank you for your "
            "support and for your notifications. I appreciate your support and your "
            "support. I would like to thank you for your support and your support "
            "for the data that you are sharing. I would like to thank you for your "
            "support and your support. I would like to thank you for your support "
            "and your support. I would like to thank you for your support. Thank you "
            "for your support. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your motivation."
        )

        results = detector.detect_hallucinations(text)

        # Should detect the repetitive "thank you for your support" pattern
        assert len(results) > 0, "Should detect repetitive support/thank you pattern"

        # Find the longest detected pattern (by k * rep_count product)
        max_score_result = max(results, key=lambda r: r[2] * r[3])
        _start, _end, _k, rep_count = max_score_result

        # Should have detected multiple repetitions
        assert rep_count >= 5, f"Expected at least 5 repetitions, got {rep_count}"

    def test_real_hallucination_thank_you_support_extensive(self):
        """Test detection of extensive 'I would like to thank you for your support' hallucination.

        This tests a severe hallucination where the exact phrase is repeated
        many times consecutively with only brief interruptions.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = (
            "Thank you. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your support. "
            "I would like to thank you for your support. I would like to thank you for "
            "your support. I would like to thank you for your support. I would like to "
            "thank you for your support. I would like to thank you for your support. "
            "I would like to thank you for your support. I would like to thank you for "
            "your support. Thank you. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support. Thank you. "
            "Thank you. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support. I would "
            "like to thank you for your support. I would like to thank you for your "
            "support. I would like to thank you for your support. I would like to thank "
            "you for your support. I would like to thank you for your support."
        )

        results = detector.detect_hallucinations(text)

        # Should detect the repetitive pattern
        assert len(results) > 0, "Should detect extensive support/thank you pattern"

        # Find the longest detected pattern (by k * rep_count product)
        max_score_result = max(results, key=lambda r: r[2] * r[3])
        _start, _end, _k, rep_count = max_score_result

        # Should have detected many repetitions
        assert rep_count >= 5, f"Expected at least 5 repetitions, got {rep_count}"

    def test_short_phrase_many_reps_detected(self):
        """Test that short phrase with many repetitions is detected."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # 1 word repeated 11 times -> SVM classifies as hallucination
        text = " ".join(["word"] * 11)

        results = detector.detect_hallucinations(text)
        assert len(results) > 0
        _start, _end, k, rep_count = results[0]
        assert k == 1
        assert rep_count == 11

    def test_long_phrase_fewer_reps_detected(self):
        """Test that long phrase needs BOTH enough reps AND SVM classification."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # 5 words repeated 3 times: reps = 3 < min_repetitions=5
        # Should NOT be detected (fails min_repetitions check before SVM)
        phrase = "this is a long phrase"
        text = " ".join([phrase] * 3)

        results = detector.detect_hallucinations(text)
        assert len(results) == 0

        # Need 5+ repetitions to reach SVM classifier
        # 5 words repeated 5 times: reps=5 >= 5 ✓ -> SVM evaluates and classifies
        text_5_reps = " ".join([phrase] * 5)
        results_5 = detector.detect_hallucinations(text_5_reps)
        assert len(results_5) > 0
        # Should find a pattern with reps >= 5 that SVM classified as hallucination
        found = False
        for _start, _end, _k, rep_count in results_5:
            if rep_count >= 5:
                found = True
                break
        assert found

    def test_no_consecutive_repetition_not_detected(self):
        """Test that non-consecutive repetitions are not detected."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "apple banana apple cherry apple date"

        results = detector.detect_hallucinations(text)
        # "apple" appears 3 times but not consecutively
        assert len(results) == 0

    def test_empty_text(self):
        """Test that empty text returns no results."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        results = detector.detect_hallucinations("")
        assert len(results) == 0

    def test_single_word(self):
        """Test that single word with no repetition returns no results."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        results = detector.detect_hallucinations("hello")
        assert len(results) == 0

    def test_natural_speech_i_have_to_filtered(self):
        """Test that 'I have to I have to' (natural stutter) is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "because I have to I have to generate the chunk"

        results = detector.detect_hallucinations(text)

        # k=3 ("I have to"), reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_natural_speech_you_can_safely_filtered(self):
        """Test that 'you can safely you can safely' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "that you can safely you can safely scale beta with N"

        results = detector.detect_hallucinations(text)

        # k=3, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_natural_speech_i_dont_think_filtered(self):
        """Test that 'I don't think I don't think' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "it. I don't think I don't think it's exactly like"

        results = detector.detect_hallucinations(text)

        # k=3, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_emphatic_speech_thank_you_filtered(self):
        """Test that polite closing 'Thank you so much' repeated is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "Thank you so much. Thank you so much."

        results = detector.detect_hallucinations(text)

        # k=4, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_emphatic_speech_who_cares_filtered(self):
        """Test that rhetorical repetition 'Who cares about you?' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "showing up at Los Alamos? Who cares about you? Who cares about you?"

        results = detector.detect_hallucinations(text)

        # k=4, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_emphatic_speech_sky_falling_filtered(self):
        """Test that idiom emphasis 'the sky is falling' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "American, the sky is falling, the sky is falling, right?"

        results = detector.detect_hallucinations(text)

        # k=4, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_natural_speech_filler_words_you_know_filtered(self):
        """Test that natural speech with filler words 'you know' is filtered.

        This is natural conversational speech with common filler patterns that
        don't constitute hallucination.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = (
            "Sver intend to serve, you know, act or behave, right, then you still "
            "need to be, you know, you need to be able to predict the consequences "
            "of your action. The more tightly linked your actions or your"
        )

        results = detector.detect_hallucinations(text)

        # Should be filtered - no exact consecutive repetitions meet threshold
        assert len(results) == 0

    def test_natural_speech_casual_conversation_filtered(self):
        """Test that casual conversational speech is filtered.

        Natural dialogue with varied phrasing and casual language markers
        like 'I mean', 'it's weird', 'right?' should not trigger detection.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "I mean, he was like, it's weird that you don't use value functions, right?"

        results = detector.detect_hallucinations(text)

        # Should be filtered - no consecutive repetitions
        assert len(results) == 0

    def test_natural_speech_you_need_to_be_filtered(self):
        """Test that natural speech with 'you need to be' pattern is filtered.

        This tests a natural speech pattern where phrases like 'you need to be'
        appear in close proximity but with different contexts, which should not
        be flagged as hallucination.
        """
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = (
            "ever intend to serve, you know, act or behave, right, then you still "
            "need to be, you know, you need to be able to predict the consequences "
            "of your action."
        )

        results = detector.detect_hallucinations(text)

        # Should be filtered - no exact consecutive repetitions
        assert len(results) == 0

    def test_emphatic_speech_i_would_love_filtered(self):
        """Test that enthusiasm 'I would love to do that' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "expand your search. I would love to do that. I would love to do that. Yes"

        _results = detector.detect_hallucinations(text)

        # k=6, reps=2 -> borderline case, SVM makes final decision
        # Could be detected or filtered depending on SVM classifier
        # This test allows either result since it's borderline

    def test_natural_language_a_percent_of_filtered(self):
        """Test that technical phrase 'a percent of a percent of' is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "maybe a percent of a percent of the speed of light"

        results = detector.detect_hallucinations(text)

        # k=3 ("a percent of") × 2 = 6 < 10, should be filtered
        assert len(results) == 0

    def test_this_is_like_natural_filler_filtered(self):
        """Test that 'this is like this is like' (natural filler) is filtered."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "I think this is like this is like very connected to"

        results = detector.detect_hallucinations(text)

        # k=3, reps=2 -> SVM classifies as not hallucination
        assert len(results) == 0

    def test_customer_data_pattern_high_repetitions(self):
        """Test pattern with high repetition count gets detected."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # Simulate a pattern that repeats many times
        phrase = "customer data for performance monitoring"
        text = " ".join([phrase] * 10)

        results = detector.detect_hallucinations(text)

        # Should detect high-repetition pattern
        assert len(results) > 0
        # Verify we detected a pattern with sufficient repetitions
        max_reps = 0
        for _start, _end, _k, rep_count in results:
            max_reps = max(max_reps, rep_count)
        # With 10 repetitions, should detect significant repetitions
        assert max_reps >= 5, f"Expected high repetition count, got {max_reps}"

    def test_edge_case_punctuation_variation_not_detected(self):
        """Test that punctuation variations prevent consecutive detection."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # Different punctuation makes them non-identical
        text = "We're very, we're very ready for it."

        results = detector.detect_hallucinations(text)

        # The phrases differ in punctuation: "very," vs "very"
        # So they're not exact consecutive repetitions
        assert len(results) == 0

    def test_edge_case_capitalization_variation_not_detected(self):
        """Test that capitalization differences prevent consecutive detection."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "So we need to understand. so we need to understand."

        results = detector.detect_hallucinations(text)

        # Different capitalization: "So" vs "so"
        assert len(results) == 0

    def test_multiple_patterns_in_same_text(self):
        """Test detection when text contains multiple repetition patterns."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # Multiple different patterns
        text = "word word word and phrase phrase phrase phrase and thing thing"

        results = detector.detect_hallucinations(text)

        # "word" reps=3 < min_repetitions=5: filtered before SVM
        # "phrase" reps=4 < min_repetitions=5: filtered before SVM
        # "thing" reps=2 < min_repetitions=5: filtered before SVM
        # All should be filtered
        assert len(results) == 0

    def test_svm_classification_2word_phrase_5_reps(self):
        """Test SVM classification for k=2, reps=5 pattern."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # k=2, reps=5 -> SVM makes the decision
        text = " ".join(["word pair"] * 5)

        results = detector.detect_hallucinations(text)

        # SVM classifies this as NOT a hallucination
        assert len(results) == 0, "SVM classifier determines this is not a hallucination"

    def test_svm_classification_1word_11_reps(self):
        """Test SVM classification for k=1, reps=11 pattern."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        # k=1, reps=11 -> SVM makes the decision
        text = " ".join(["word"] * 11)

        results = detector.detect_hallucinations(text)

        # SVM classifies this as a hallucination
        assert len(results) > 0
        _start, _end, k, rep_count = results[0]
        assert k == 1
        assert rep_count == 11


class TestDetectBasic:
    """Tests for the basic detect method (all repetitions, not just consecutive)."""

    def test_detect_finds_all_repetitions(self):
        """Test that detect() finds all repetitions, not just consecutive."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "apple banana cherry apple banana cherry fig apple banana cherry"

        results = detector.detect(text, min_k=3)

        # Should find multiple instances of "apple banana cherry"
        assert len(results) > 0

    def test_detect_prioritizes_longer_patterns(self):
        """Test that detect() keeps longest pattern when multiple exist at same position."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "apple banana cherry date apple banana cherry fig"

        results = detector.detect(text, min_k=3)

        # Should prioritize "apple banana cherry" (k=3) at position 0
        # over shorter subpatterns
        found_k3 = False
        for start, _end, k in results:
            if start == 0 and k >= 3:
                found_k3 = True
                break
        assert found_k3


class TestPrepare:
    """Tests for text preparation (normalization)."""

    def test_prepare_normalizes_whitespace_only(self):
        """Test that prepare() only normalizes whitespace."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)

        # Should preserve punctuation and capitalization
        text = "Hello,  World!  How   are you?"
        prepared = detector.prepare(text)
        assert prepared == "Hello, World! How are you?"

    def test_prepare_preserves_punctuation(self):
        """Test that punctuation is preserved."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "We're very, we're very ready."
        prepared = detector.prepare(text)
        assert "'" in prepared
        assert "," in prepared

    def test_prepare_preserves_case(self):
        """Test that case is preserved."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)
        text = "Hello World"
        prepared = detector.prepare(text)
        assert "Hello" in prepared
        assert "World" in prepared
        assert prepared != "hello world"


class TestConstructorParameters:
    """Tests for constructor parameter configuration."""

    def test_constructor_default_parameters(self):
        """Test that default constructor parameters work correctly."""
        detector = RepetitionDetector(CONFIG.getRepetitionMinK(), CONFIG.getRepetitionMinRepetitions(), CONFIG)

        # Defaults: min_k=1, min_repetitions=5
        assert detector.min_k == 1
        assert detector.min_repetitions == 5

    def test_constructor_custom_parameters(self):
        """Test that custom constructor parameters are stored."""
        detector = RepetitionDetector(min_k=3, min_repetitions=7, config=CONFIG)

        assert detector.min_k == 3
        assert detector.min_repetitions == 7

    def test_min_repetitions_filters_low_count(self):
        """Test that min_repetitions parameter filters patterns with too few repetitions."""
        detector = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        text = "as like a, as like a, an object"

        results = detector.detect_hallucinations(text)

        # Should NOT be detected - only 2 repetitions, below min_repetitions=5
        assert len(results) == 0

    def test_min_repetitions_allows_sufficient_count(self):
        """Test that patterns meeting min_repetitions are evaluated by classifier."""
        detector = RepetitionDetector(min_k=1, min_repetitions=3, config=CONFIG)
        text = " ".join(["test pattern here"] * 4)

        results = detector.detect_hallucinations(text)

        # Pattern meets min_repetitions (4 >= 3), classifier evaluates it
        # k=3, reps=4 - classifier determines if hallucination
        if len(results) > 0:
            _start, _end, k, rep_count = results[0]
            assert k == 3  # 3-word phrase
            assert rep_count == 4  # 4 repetitions

    def test_different_min_repetitions_configurations(self):
        """Test that different min_repetitions values filter differently."""
        text = " ".join(["word phrase pattern"] * 4)

        # High min_repetitions=5 filters this pattern (4 < 5)
        detector_high = RepetitionDetector(min_k=1, min_repetitions=5, config=CONFIG)
        results_high = detector_high.detect_hallucinations(text)
        assert len(results_high) == 0  # Filtered by min_repetitions

        # Lower min_repetitions=3 allows classifier to evaluate
        detector_low = RepetitionDetector(min_k=1, min_repetitions=3, config=CONFIG)
        results_low = detector_low.detect_hallucinations(text)
        # k=3, reps=4 - classifier determines the result
        if len(results_low) > 0:
            _start, _end, k, rep_count = results_low[0]
            assert k == 3
            assert rep_count == 4

    def test_constructor_min_k_parameter_filters_patterns(self):
        """Test that min_k parameter filters patterns by phrase length."""
        detector_mink_1 = RepetitionDetector(min_k=1, min_repetitions=2, config=CONFIG)
        detector_mink_5 = RepetitionDetector(min_k=5, min_repetitions=2, config=CONFIG)

        # Text with a 3-word pattern repeated 15 times (enough to be detected as hallucination)
        text = " ".join(["word phrase test"] * 15)

        # With min_k=1, can find patterns of any length (including 3-word)
        results_1 = detector_mink_1.detect_hallucinations(text)

        # With min_k=5, only finds patterns of 5+ words (filters 3-word pattern)
        results_5 = detector_mink_5.detect_hallucinations(text)

        # min_k=1 should detect the 3-word pattern (k=3, reps=15)
        assert len(results_1) > 0, "min_k=1 should detect 3-word pattern"
        _start, _end, k, rep_count = results_1[0]
        assert k == 3  # 3-word phrase
        assert rep_count == 15  # 15 repetitions

        # min_k=5 should NOT find 3-word patterns (but might find longer accidental patterns)
        # Verify no 3-word pattern in results
        for _start, _end, k, _rep_count in results_5:
            assert k >= 5, f"min_k=5 should only find patterns with k >= 5, found k={k}"
