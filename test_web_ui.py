#!/usr/bin/env python3
"""Regression tests for the Flask web UI error handling."""

import cadquery as cq
import pytest

import web_ui


@pytest.fixture
def client(tmp_path):
    original_parts = list(web_ui.state.parts)
    original_dir = web_ui.state.parts_dir
    original_workflow = web_ui.state.workflow
    original_cache_dir = web_ui._WEBUI_CACHE_DIR
    original_mesh_cache = dict(web_ui._MESH_PAYLOAD_CACHE)
    original_decimated_cache = dict(web_ui._DECIMATED_PAYLOAD_CACHE)
    web_ui.state.parts = []
    web_ui.state.parts_dir = tmp_path
    web_ui.state.workflow = "cylinder"
    web_ui._WEBUI_CACHE_DIR = tmp_path / ".webui_cache"
    web_ui._MESH_PAYLOAD_CACHE.clear()
    web_ui._DECIMATED_PAYLOAD_CACHE.clear()
    web_ui.app.config["TESTING"] = True
    try:
        with web_ui.app.test_client() as client:
            yield client, tmp_path
    finally:
        web_ui.state.parts = original_parts
        web_ui.state.parts_dir = original_dir
        web_ui.state.workflow = original_workflow
        web_ui._WEBUI_CACHE_DIR = original_cache_dir
        web_ui._MESH_PAYLOAD_CACHE.clear()
        web_ui._MESH_PAYLOAD_CACHE.update(original_mesh_cache)
        web_ui._DECIMATED_PAYLOAD_CACHE.clear()
        web_ui._DECIMATED_PAYLOAD_CACHE.update(original_decimated_cache)


def test_load_invalid_step_returns_json_error_and_preserves_state(client):
    client, tmp_path = client

    # Seed existing state to verify failed loads do not wipe the session.
    good_shape = cq.Workplane("XY").box(10, 10, 10).val().wrapped
    web_ui.state.parts = [
        web_ui.PartState(
            file_path=str(tmp_path / "existing.step"),
            name="existing",
            source_ext=".step",
            shape=good_shape,
        )
    ]

    bad_file = tmp_path / "broken.step"
    bad_file.write_text("this is not a valid STEP file", encoding="ascii")

    resp = client.post("/api/load", json={"files": ["broken.step"]})

    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "failed to load" in data["error"]
    assert data["file"] == "broken.step"
    assert len(web_ui.state.parts) == 1
    assert web_ui.state.parts[0].name == "existing"


def test_upload_rejects_missing_file_field(client):
    client, _tmp_path = client
    resp = client.post("/api/upload-part", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert data["error"] == "missing file"


def test_scene_reuses_cached_preview_payload_for_combined_entries(client, monkeypatch):
    client, tmp_path = client

    mesh_path = tmp_path / "box.stl"
    cq.exporters.export(cq.Workplane("XY").box(10, 10, 10), str(mesh_path))
    wp, _name = web_ui._load_cached_part(str(mesh_path), require_solid=False)
    web_ui.state.parts = [
        web_ui.PartState(
            file_path=str(mesh_path),
            name="outer_1",
            source_ext=".stl",
            shape=wp.val().wrapped,
            mesh_source_path=str(mesh_path),
        )
    ]

    first_scene = web_ui._build_scene()
    assert len(first_scene["parts"]) == 1
    assert first_scene["parts"][0]["meshSourcePath"] == str(mesh_path)
    assert first_scene["combined"][0]["partIndex"] == 0
    assert "mesh" not in first_scene["combined"][0]

    cache_files = list((tmp_path / ".webui_cache" / "mesh_payloads").glob("*.json.gz"))
    assert cache_files

    def fail_mesh_payload(_shape, tolerance=0.5):
        raise AssertionError(f"unexpected tessellation at tolerance {tolerance}")

    monkeypatch.setattr(web_ui, "_mesh_payload", fail_mesh_payload)
    second_scene = web_ui._build_scene()
    assert second_scene["combined"][0]["partIndex"] == 0
    assert second_scene["parts"][0]["mesh"]["vertices"]


def test_debug_log_endpoint_accepts_client_entries(client):
    client, _tmp_path = client

    resp = client.post("/api/debug-log", json={
        "entries": [
            {"ts": 123, "kind": "api-start", "message": "GET /api/scene", "meta": {"a": 1}},
            {"ts": 124, "kind": "stall-warning", "message": "load-selected has no new progress for 5s"},
        ]
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["accepted"] == 2

    snapshot = client.get("/api/debug-log")
    assert snapshot.status_code == 200
    payload = snapshot.get_json()
    assert payload is not None
    messages = [entry["message"] for entry in payload["entries"]]
    assert "GET /api/scene" in messages
    assert "load-selected has no new progress for 5s" in messages


def test_auto_orient_stage_uses_decimated_preview_meshes(client):
    client, tmp_path = client

    step_path = tmp_path / "outer_1.step"
    cq.exporters.export(cq.Workplane("XY").cylinder(20, 8), str(step_path))

    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    resp = client.post("/api/stage/auto_orient", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "decimated preview meshes" in data["message"]
    assert web_ui.state.parts[0].auto_oriented is True
    assert web_ui.state.parts[0].orientation_steps
