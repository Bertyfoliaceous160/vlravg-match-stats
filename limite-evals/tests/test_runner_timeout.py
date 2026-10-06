from limite_evals.runner import _client_timeout


def test_client_timeout_leaves_only_response_reads_unbounded() -> None:
    timeout = _client_timeout(1800.0)

    assert timeout.connect == 1800.0
    assert timeout.write == 1800.0
    assert timeout.pool == 1800.0
    assert timeout.read is None
