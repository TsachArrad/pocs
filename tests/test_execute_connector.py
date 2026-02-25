import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _load_fixture() -> dict:
    fixture_path = Path(__file__).parent / "fixtures" / "python_connector.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def _load_real_connector_fixture() -> dict:
    custom_file = os.getenv("CONNECTOR_TEST_JSON_FILE")
    if custom_file:
        custom_path = Path(custom_file)
        if not custom_path.exists():
            raise FileNotFoundError(f"CONNECTOR_TEST_JSON_FILE not found: {custom_path}")
        return json.loads(custom_path.read_text(encoding="utf-8"))

    default_path = Path(__file__).parent / "fixtures" / "local_real_connector.json"
    return json.loads(default_path.read_text(encoding="utf-8"))


def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def test_execute_connector_success_with_mocked_docker(monkeypatch):
    connector = _load_fixture()

    def fake_run(command, capture_output, text, timeout, check):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "ok": True,
                    "result": {
                        "spreadsheetId": "abc123",
                        "sheetName": "Sheet1",
                        "columnIndex": 3,
                        "columnLetter": "C",
                    },
                }
            ),
            stderr="",
        )

    monkeypatch.setattr("app.main.subprocess.run", fake_run)

    client = TestClient(app)
    response = client.post(
        "/execute-connector",
        json={
            "connector": connector,
            "env": {
                "CONNECTOR_JSON": "{}"
            },
            "network_mode": "bridge",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"]["columnLetter"] == "C"


def test_execute_connector_missing_required_path_returns_400(monkeypatch):
    connector = _load_fixture()
    del connector["configuration"]["sheetName"]

    client = TestClient(app)
    response = client.post(
        "/execute-connector",
        json={
            "connector": connector,
            "env": {"CONNECTOR_JSON": "{}"},
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert "missing" in body["detail"]


@pytest.mark.integration
def test_execute_connector_real_docker_run():
    if not _docker_available():
        pytest.skip("Docker is not available for real integration execution")

    connector = _load_real_connector_fixture()

    client = TestClient(app)
    response = client.post(
        "/execute-connector",
        json={
            "connector": connector,
            "env": {
                "TEST_SUFFIX": "from-real-test",
            },
            "network_mode": "none",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["result"]["message"] == "hello-from-real-test"


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([
            "-m",
            "integration",
            "tests/test_execute_connector.py::test_execute_connector_real_docker_run",
            "-q",
        ])
    )
