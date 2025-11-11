import pytest
from pytest import MonkeyPatch
from unittest.mock import MagicMock, Mock, mock_open
from httpx import HTTPStatusError, RequestError

from tests.llamaserve import TestLlamaServe


@pytest.fixture
def server() -> TestLlamaServe:
    server: TestLlamaServe = TestLlamaServe()
    return server


@pytest.fixture
def mock_httpx_get(monkeypatch: MonkeyPatch) -> Mock:
    mock: Mock = Mock(return_value=Mock(json=Mock(return_value={'url': 'foobar'})))
    monkeypatch.setattr('httpx.get', mock)
    return mock


@pytest.fixture
def mock_httpx_stream(monkeypatch: MonkeyPatch) -> MagicMock:
    chunks = [b'foo', b'bar', b'baz']
    mock: MagicMock = MagicMock(
        raise_for_status=MagicMock(return_value=None),
        headers={'content-length': str(sum(len(chunk) for chunk in chunks))},
        iter_bytes=MagicMock(return_value=chunks),
    )
    monkeypatch.setattr("httpx.stream", mock)
    mock_pbar = MagicMock()
    monkeypatch.setattr("tqdm.tqdm", mock_pbar)
    return mock


@pytest.fixture
def mock_file(monkeypatch: MonkeyPatch) -> Mock:
    mock: Mock = mock_open()
    monkeypatch.setattr('builtins.open', mock)
    return mock


def test_get_weights(
    monkeypatch: MonkeyPatch,
    server: TestLlamaServe,
    mock_httpx_get: Mock,
    mock_httpx_stream: MagicMock,
    mock_file: Mock,
) -> None:
    monkeypatch.setattr('os.path.isfile', Mock(return_value=False))
    server.get_weights('foo')
    mock_httpx_get.assert_any_call(
        server.get_presigned_generation_url(), headers={'x-api-key': 'foo'}
    )
    mock_httpx_stream.assert_any_call('GET', 'foobar')
    mock_file.assert_called_once_with(server.get_weights_path(), 'wb')


def test_api_key(
    monkeypatch: MonkeyPatch, server: TestLlamaServe, mock_httpx_get: Mock
) -> None:
    monkeypatch.setattr('os.path.isfile', Mock(return_value=False))
    mock_httpx_get.side_effect = HTTPStatusError(
        '401 Unauthorized', request=Mock(), response=Mock(status_code=401)
    )
    assert not server.get_weights('foo')


def test_exceptions(
    monkeypatch: MonkeyPatch, server: TestLlamaServe, mock_httpx_get: Mock
) -> None:
    monkeypatch.setattr('os.path.isfile', Mock(return_value=False))
    mock_httpx_get.side_effect = RequestError('Network failure')
    assert not server.get_weights('foo')


def test_weights_exist(
    monkeypatch: MonkeyPatch, server: TestLlamaServe, mock_httpx_get: Mock
) -> None:
    monkeypatch.setattr('os.path.isfile', Mock(return_value=True))
    assert server._get_weights('foo')
    mock_httpx_get.assert_not_called()
