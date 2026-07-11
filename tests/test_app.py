"""App surface tests -- no lifespan, no models, no network.

TestClient used WITHOUT a `with` block never runs the lifespan, so the app
stays not-ready: exactly the state we can assert against offline.
"""
import json

from fastapi.testclient import TestClient

from visawise.app.main import app
from visawise.config import settings

client = TestClient(app, raise_server_exceptions=False)


def test_health_503_before_ready():
    response = client.get("/health")
    assert response.status_code == 503


def test_chat_503_before_ready():
    response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503


def test_chat_validation_rejects_bad_payloads():
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 4001}).status_code == 422
    assert client.post("/api/chat", json={"message": "hi", "extra": 1}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422


def test_eval_runs_404_without_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    response = client.get("/api/eval/runs")
    assert response.status_code == 404


def test_eval_runs_serves_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "runs_dir", tmp_path)
    summary = {"generated_at": "2026-07-11T00:00:00+00:00", "runs": [{"run_id": "r1"}]}
    (tmp_path / "latest_summary.json").write_text(json.dumps(summary))

    response = client.get("/api/eval/runs")
    assert response.status_code == 200
    assert response.json() == summary
