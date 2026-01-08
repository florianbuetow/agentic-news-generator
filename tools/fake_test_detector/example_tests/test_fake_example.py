"""Example of fake unit tests for validation.

These tests are deliberately fake and should be detected by the fake test detector tool.
"""


def test_fake_just_passes():
    """This test is fake - just passes without testing anything."""
    pass


def test_fake_no_assertions():
    """This test sets up objects but doesn't assert anything."""
    from unittest.mock import Mock

    mock_obj = Mock()
    mock_obj.method = Mock(return_value=True)
    # No assertions - just calls a mock


def test_fake_hardcoded_mock():
    """This test only tests the mock, not actual logic."""
    from unittest.mock import Mock

    mock_service = Mock()
    mock_service.get_data = Mock(return_value={"key": "hardcoded value"})

    # Only testing that the mock returns what we told it to return
    result = mock_service.get_data()
    assert result == {"key": "hardcoded value"}


def test_fake_trivial_assertion():
    """This test has trivial assertions that don't test real behavior."""
    x = 1 + 1
    assert True
    assert 1 == 1


def test_fake_exception_swallowing():
    """This test swallows exceptions and hides failures."""
    try:
        # This would normally fail but we hide it
        result = some_function_that_should_work()
        assert result == "expected"
    except Exception:
        pass  # Swallow all exceptions - test always passes


def test_fake_commented_assertions():
    """This test has commented-out assertions."""
    data = {"key": "value"}
    result = process_data(data)
    # assert result is not None
    # assert "processed" in result
    # assert result["status"] == "success"


def test_fake_never_calls_function():
    """This test sets up but never calls the function under test."""
    from unittest.mock import Mock

    # Setup
    mock_db = Mock()
    mock_logger = Mock()

    # Never actually call the function we're supposed to test
    # service = UserService(mock_db, mock_logger)
    # service.create_user("test")

    # Just assert on the mocks
    assert mock_db is not None


def test_fake_only_comments():
    """This test is just comments about what should be tested."""
    # TODO: Test that the user creation works
    # TODO: Verify email is sent
    # TODO: Check database is updated


def test_fake_mock_call_assertions():
    """This test only checks mock calls, not actual behavior."""
    from unittest.mock import Mock

    mock_api = Mock()
    mock_api.send_request = Mock(return_value=True)

    mock_api.send_request("data")

    # Only asserting the mock was called, not testing real logic
    mock_api.send_request.assert_called_once_with("data")


def test_fake_mock_assertions_only():
    """This test asserts on mock behavior, not business logic."""
    from unittest.mock import Mock, call

    mock_service = Mock()
    processor = DataProcessor(mock_service)

    processor.process([1, 2, 3])

    # Only checking mock interactions, not actual results
    assert mock_service.handle.call_count == 3
    assert call(1) in mock_service.handle.call_args_list
