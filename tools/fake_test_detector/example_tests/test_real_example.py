"""Example of real unit tests for validation.

These tests are legitimate and should NOT be detected as fake.
"""


def test_addition_with_assertions():
    """Real test - tests addition with proper assertions."""
    result = 2 + 2
    assert result == 4
    assert isinstance(result, int)


def test_string_operations():
    """Real test - tests string operations."""
    text = "hello world"
    result = text.upper()
    assert result == "HELLO WORLD"
    assert len(result) == 11


def test_list_operations():
    """Real test - tests list operations."""
    items = [1, 2, 3]
    items.append(4)
    assert len(items) == 4
    assert items[-1] == 4
    assert 2 in items
