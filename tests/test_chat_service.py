from unittest.mock import MagicMock, patch

import pytest

from app.llm.chat_service import _extract_retry_delay, _is_rate_limit_error, complete


def make_rate_limit_error(retry_delay: str = "55s"):
    error = Exception("429 RESOURCE_EXHAUSTED")
    error.code = 429
    error.details = {
        "error": {
            "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}
            ]
        }
    }
    return error


def test_is_rate_limit_error_true_for_429():
    assert _is_rate_limit_error(make_rate_limit_error())


def test_is_rate_limit_error_false_for_other_errors():
    error = Exception("boom")
    error.code = 500
    assert not _is_rate_limit_error(error)


def test_extract_retry_delay_parses_seconds_from_retry_info():
    assert _extract_retry_delay(make_rate_limit_error("55s")) == 55.0


def test_extract_retry_delay_returns_none_when_details_are_missing():
    error = Exception("boom")
    assert _extract_retry_delay(error) is None


@patch("app.llm.chat_service.time.sleep")
@patch("app.llm.chat_service.client")
def test_complete_retries_after_rate_limit_then_succeeds(mock_client, mock_sleep):
    success_response = MagicMock(text="the answer")
    mock_client.models.generate_content.side_effect = [
        make_rate_limit_error("1s"),
        success_response,
    ]

    result = complete("a prompt")

    assert result == "the answer"
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("app.llm.chat_service.time.sleep")
@patch("app.llm.chat_service.client")
def test_complete_gives_up_after_max_retries(mock_client, mock_sleep):
    mock_client.models.generate_content.side_effect = make_rate_limit_error("1s")

    with pytest.raises(Exception):
        complete("a prompt", max_retries=3)

    assert mock_client.models.generate_content.call_count == 3


@patch("app.llm.chat_service.time.sleep")
@patch("app.llm.chat_service.client")
def test_complete_does_not_retry_non_rate_limit_errors(mock_client, mock_sleep):
    error = Exception("500 internal error")
    error.code = 500
    mock_client.models.generate_content.side_effect = error

    with pytest.raises(Exception):
        complete("a prompt")

    assert mock_client.models.generate_content.call_count == 1
    mock_sleep.assert_not_called()
