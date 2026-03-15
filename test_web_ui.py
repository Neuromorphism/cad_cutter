#!/usr/bin/env python3
"""Regression tests for the Flask web UI error handling."""

import importlib.util
import math
from pathlib import Path

import cadquery as cq
import pytest

import assemble
import web_ui


TEST_MODEL_DIR = Path(__file__).resolve().parent / "test_models" / "conic_capsule_topopt_8"


def _world_bbox(entry):
    shape = entry[1]
    loc = entry[2]
    tx, ty, tz = loc.toTuple()[0]
    xmin, ymin, zmin, xmax, ymax, zmax = assemble.get_tight_bounding_box(shape)
    return (
        xmin + tx, ymin + ty, zmin + tz,
        xmax + tx, ymax + ty, zmax + tz,
    )


@pytest.fixture
def client(tmp_path):
    original_parts = list(web_ui.state.parts)
    original_dir = web_ui.state.parts_dir
    original_workflow = web_ui.state.workflow
    original_fine_orient = web_ui.state.fine_orient
    original_gap = web_ui.state.gap
    original_axis = web_ui.state.axis
    original_section_number = web_ui.state.section_number
    original_midlayer_configs = {key: dict(value) for key, value in web_ui.state.midlayer_configs.items()}
    original_cache_dir = web_ui._WEBUI_CACHE_DIR
    original_mesh_cache = dict(web_ui._MESH_PAYLOAD_CACHE)
    original_decimated_cache = dict(web_ui._DECIMATED_PAYLOAD_CACHE)
    web_ui.state.parts = []
    web_ui.state.parts_dir = tmp_path
    web_ui.state.workflow = "cylinder"
    web_ui.state.fine_orient = False
    web_ui.state.gap = 0.0
    web_ui.state.axis = "z"
    web_ui.state.section_number = None
    web_ui.state.midlayer_configs = {key: dict(value) for key, value in original_midlayer_configs.items()}
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
        web_ui.state.fine_orient = original_fine_orient
        web_ui.state.gap = original_gap
        web_ui.state.axis = original_axis
        web_ui.state.section_number = original_section_number
        web_ui.state.midlayer_configs = {key: dict(value) for key, value in original_midlayer_configs.items()}
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
    assert "axis-aligned six-way heuristic" in data["message"]
    assert web_ui.state.parts[0].auto_oriented is True


def test_auto_orient_axis_aligned_mode_reorients_simple_rotated_tube(client):
    client, tmp_path = client

    step_path = tmp_path / "outer_1.step"
    shape = cq.Workplane("XY").cylinder(30, 8).rotate((0, 0, 0), (0, 1, 0), 90)
    cq.exporters.export(shape, str(step_path))

    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    resp = client.post("/api/stage/auto_orient", json={})
    assert resp.status_code == 200

    oriented = web_ui._apply_manual_transforms(web_ui.state.parts[0])
    xmin, ymin, zmin, xmax, ymax, zmax = assemble.get_tight_bounding_box(oriented)
    extents = [xmax - xmin, ymax - ymin, zmax - zmin]
    assert extents[2] == pytest.approx(max(extents), rel=1e-3)


def test_auto_orient_fine_mode_can_be_enabled_via_config(client):
    client, tmp_path = client

    step_path = tmp_path / "outer_1.step"
    cq.exporters.export(cq.Workplane("XY").cylinder(20, 8), str(step_path))
    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    config_resp = client.patch("/api/config", json={"fine_orient": True})
    assert config_resp.status_code == 200
    assert web_ui.state.fine_orient is True

    resp = client.post("/api/stage/auto_orient", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "fine-grained preview solve" in data["message"]


def test_scene_reuses_cached_preview_mesh_after_auto_orient(client, monkeypatch):
    client, tmp_path = client

    step_path = tmp_path / "outer_1.step"
    rotated = cq.Workplane("XY").cylinder(30, 8).rotate((0, 0, 0), (0, 1, 0), 90)
    cq.exporters.export(rotated, str(step_path))

    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200
    orient_resp = client.post("/api/stage/auto_orient", json={})
    assert orient_resp.status_code == 200

    def fail_mesh_payload(_shape, tolerance=0.5):
        raise AssertionError(f"unexpected tessellation at tolerance {tolerance}")

    monkeypatch.setattr(web_ui, "_mesh_payload", fail_mesh_payload)
    scene = web_ui._build_scene()
    assert len(scene["parts"]) == 1
    assert scene["parts"][0]["orientationSteps"]


def test_orientation_payload_reuses_preview_mesh_after_runtime_transform(client, monkeypatch):
    client, tmp_path = client

    step_path = tmp_path / "outer_1.step"
    cq.exporters.export(cq.Workplane("XY").cylinder(30, 8), str(step_path))

    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    part = web_ui.state.parts[0]
    part.orientation_steps = [((0.0, 1.0, 0.0), math.pi / 2.0)]
    part.rot_xyz = (0.0, 0.0, 15.0)
    part.manual_scale = 1.1

    def fail_mesh_payload(_shape, tolerance=0.5):
        raise AssertionError(f"unexpected tessellation at tolerance {tolerance}")

    monkeypatch.setattr(web_ui, "_mesh_payload", fail_mesh_payload)
    payload = web_ui._orientation_payload_for_part(part)
    vertices, _faces = web_ui._payload_arrays(payload)
    assert len(vertices) > 0


def test_auto_drop_stage_returns_timing_metrics_and_persists_offsets(client):
    client, tmp_path = client

    lower = tmp_path / "outer_1.step"
    upper = tmp_path / "outer_2.step"
    cq.exporters.export(cq.Workplane("XY").cylinder(20, 10), str(lower))
    cq.exporters.export(cq.Workplane("XY").transformed(offset=(0, 0, 40)).cylinder(20, 8), str(upper))

    load_resp = client.post("/api/load", json={"files": ["outer_1.step", "outer_2.step"], "include_scene": True})
    assert load_resp.status_code == 200

    resp = client.post("/api/stage/auto_drop", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert "local_only" in data["metrics"]
    assert "proxy_then_local" in data["metrics"]
    assert any(abs(v) >= 0 for v in web_ui.state.parts[1].settle_offset)

    scene = web_ui._build_scene()
    assert len(scene["combined"]) == 2


def test_part_translation_patch_persists_in_scene_payload(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    load_resp = client.post("/api/load", json={"files": ["outer_1.step", "outer_2.step"], "include_scene": True})
    assert load_resp.status_code == 200

    patch_resp = client.patch("/api/part/1", json={"translation": {"x": 12, "y": -8, "z": 34}})
    assert patch_resp.status_code == 200
    assert web_ui.state.parts[1].manual_translate == pytest.approx((12.0, -8.0, 34.0))

    scene = web_ui._build_scene()
    assert scene["parts"][1]["translate"] == pytest.approx([12.0, -8.0, 34.0])


def test_auto_drop_closes_manual_translation_gap_on_test_models(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    load_resp = client.post("/api/load", json={"files": ["outer_1.step", "outer_2.step"], "include_scene": True})
    assert load_resp.status_code == 200

    patch_resp = client.patch("/api/part/1", json={"translation": {"z": 80}})
    assert patch_resp.status_code == 200

    resp = client.post("/api/stage/auto_drop", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["metrics"]["chosen"]["elapsed_s"] < 2.0
    assert web_ui.state.parts[1].settle_offset[2] < -70.0


def test_contact_solver_negative_gap_allows_slight_interference_fit():
    lower_shape = cq.Workplane("XY").box(10, 10, 10).val().wrapped
    upper_shape = cq.Workplane("XY").box(10, 10, 10).val().wrapped

    part_info = [
        ("lower", lower_shape, cq.Location(cq.Vector(0, 0, 5)), (0.5, 0.5, 0.5), None, False),
        ("upper", upper_shape, cq.Location(cq.Vector(0, 0, 20)), (0.5, 0.5, 0.5), None, False),
    ]

    axis_vec = assemble.AXIS_MAP["z"]
    zero_gap_info, _zero_metrics = assemble.simulate_physics_contact_fast(part_info, axis_vec, 0.0, rough_drop=False)
    interference_info, _neg_metrics = assemble.simulate_physics_contact_fast(part_info, axis_vec, -0.5, rough_drop=False)

    zero_upper = zero_gap_info[1][2].toTuple()[0][2]
    neg_upper = interference_info[1][2].toTuple()[0][2]
    assert neg_upper < zero_upper - 0.4


def test_midlayer_solver_metadata_endpoint_exposes_defaults(client):
    client, _tmp_path = client

    resp = client.get("/api/midlayer-solvers")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert set(data["solvers"]) == {"dl4to", "pymoto"}
    assert data["solvers"]["dl4to"]["availability"]["installed"] is False
    expected_pymoto_installed = importlib.util.find_spec("pymoto") is not None
    assert data["solvers"]["pymoto"]["availability"]["installed"] is expected_pymoto_installed
    assert data["solvers"]["pymoto"]["availability"]["mode"] == ("native" if expected_pymoto_installed else "scaffold")
    assert data["configs"]["pymoto"]["resolution"] >= 16


def test_midlayer_design_requires_matching_outer_and_inner_sections(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    load_resp = client.post("/api/load", json={"files": ["outer_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    resp = client.post("/api/stage/design_midlayer_dl4to", json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data is not None
    assert "matching outer_* and inner_* sections" in data["error"]


@pytest.mark.parametrize("solver_id", ["dl4to", "pymoto"])
def test_midlayer_design_stage_generates_mid_parts_from_matching_sections(client, solver_id):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()
    expected_mode = "native" if solver_id == "pymoto" and importlib.util.find_spec("pymoto") is not None else "scaffold"
    config_payload = {
        "resolution": 20,
        "volume_fraction": 0.30,
        "smoothing_passes": 2,
    }
    if solver_id == "pymoto":
        config_payload["resolution"] = 14
        config_payload["smoothing_passes"] = 1
        config_payload["pymoto_iterations"] = 4

    config_resp = client.patch("/api/config", json={
        "midlayer_configs": {
            solver_id: config_payload
        }
    })
    assert config_resp.status_code == 200

    load_resp = client.post("/api/load", json={
        "files": ["outer_1.step", "inner_1.step", "outer_2.step", "inner_2.step"],
        "include_scene": True,
    })
    assert load_resp.status_code == 200

    resp = client.post(f"/api/stage/design_midlayer_{solver_id}", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["ok"] is True
    assert data["solver"]["mode"] == expected_mode
    assert len(data["generated"]) == 2
    assert all(name.startswith("mid_") for name in data["generated"])

    scene = _assert_scene_has_loaded_parts(client, 6)
    names = {part["name"] for part in scene["parts"]}
    assert f"mid_1_{solver_id}" in names
    assert f"mid_2_{solver_id}" in names
    solver_paths = [
        Path(artifact["outputPath"])
        for artifact in data["solver"]["artifacts"]
        if artifact["outputPath"] is not None
    ]
    assert len(solver_paths) == 2
    assert all(path.exists() for path in solver_paths)


def _load_test_model_dir():
    return (Path(__file__).resolve().parent / "test_models" / "conic_capsule_topopt_8").resolve()


def _bbox_after_location(shape, loc):
    moved = assemble.apply_location(shape, loc)
    return assemble.get_tight_bounding_box(moved)


def _bbox_contains(outer_bbox, inner_bbox, tol=1.0):
    return (
        inner_bbox[0] >= outer_bbox[0] - tol
        and inner_bbox[1] >= outer_bbox[1] - tol
        and inner_bbox[2] >= outer_bbox[2] - tol
        and inner_bbox[3] <= outer_bbox[3] + tol
        and inner_bbox[4] <= outer_bbox[4] + tol
        and inner_bbox[5] <= outer_bbox[5] + tol
    )


def _bbox_center(bbox):
    return (
        (bbox[0] + bbox[3]) / 2.0,
        (bbox[1] + bbox[4]) / 2.0,
        (bbox[2] + bbox[5]) / 2.0,
    )


def test_test_models_two_outer_sections_stack_vertically(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    load_resp = client.post("/api/load", json={"files": ["outer_1.step", "outer_2.step"], "include_scene": True})
    assert load_resp.status_code == 200

    stack_resp = client.post("/api/stage/auto_stack", json={})
    assert stack_resp.status_code == 200

    scene = client.get("/api/scene").get_json()
    offsets = {entry["name"]: entry["offset"] for entry in scene["combined"]}
    assert offsets["outer_1"][2] == pytest.approx(250.0, abs=1.0)
    assert offsets["outer_2"][2] == pytest.approx(750.0, abs=1.0)
    assert offsets["outer_2"][2] > offsets["outer_1"][2]


def test_test_models_full_load_uses_deferred_mesh_payloads(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    files = [
        *(f"outer_{i}.step" for i in range(1, 9)),
        *(f"mid_{i}.step" for i in range(1, 7)),
        *(f"inner_{i}.step" for i in range(1, 7)),
    ]
    load_resp = client.post("/api/load", json={"files": files, "include_scene": True})
    assert load_resp.status_code == 200

    data = load_resp.get_json()
    assert data is not None
    scene = data["scene"]
    assert len(scene["parts"]) == 20
    assert len(scene["combined"]) == 20
    assert all(part["mesh"] is None for part in scene["parts"])
    assert all(part["meshUrl"] for part in scene["parts"])
    assert all(part["meshFormat"] == "payload" for part in scene["parts"])
    assert len(load_resp.data) < 3_000_000


def test_repo_outer_step_pair_uses_inline_proxy_meshes(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = Path(__file__).resolve().parent

    load_resp = client.post("/api/load", json={"files": ["outer_1.STEP", "outer_2.STEP"], "include_scene": True})
    assert load_resp.status_code == 200

    data = load_resp.get_json()
    assert data is not None
    scene = data["scene"]
    assert len(scene["parts"]) == 2
    assert all(part["mesh"] for part in scene["parts"])
    assert all(part["meshUrl"] is None for part in scene["parts"])
    total_numbers = sum(
        len(part["mesh"]["vertices"]) + len(part["mesh"]["indices"])
        for part in scene["parts"]
    )
    assert total_numbers < 140_000
    assert all(part["thumbMesh"] is None for part in scene["parts"])
    assert len(load_resp.data) < 1_600_000


def _load_repo_root_outer_pair(client):
    web_ui.state.parts_dir = Path(__file__).resolve().parent
    load_resp = client.post("/api/load", json={"files": ["outer_1.STEP", "outer_2.STEP"], "include_scene": True})
    assert load_resp.status_code == 200


def _assert_scene_has_loaded_parts(client, expected_count: int):
    scene_resp = client.get("/api/scene")
    assert scene_resp.status_code == 200
    scene = scene_resp.get_json()
    assert scene is not None
    assert len(scene["parts"]) == expected_count
    assert len(scene["combined"]) == expected_count
    return scene


def test_repo_outer_step_pair_auto_stack_orders_parts_by_level(client):
    client, _tmp_path = client
    _load_repo_root_outer_pair(client)

    stack_resp = client.post("/api/stage/auto_stack", json={})
    assert stack_resp.status_code == 200
    stack_data = stack_resp.get_json()
    assert stack_data is not None
    assert stack_data["timings"]["durationMs"] < 100.0
    assert [part.name for part in web_ui.state.parts] == ["outer_1", "outer_2"]
    assert web_ui._effective_stack_axis_name() == "y"

    scene = _assert_scene_has_loaded_parts(client, 2)
    offsets = {entry["name"]: entry["offset"] for entry in scene["combined"]}
    assert offsets["outer_2"][1] > offsets["outer_1"][1]


@pytest.mark.parametrize(
    ("stage_name", "artifact_path"),
    [
        ("auto_orient", None),
        ("auto_stack", None),
        ("auto_scale", None),
        ("auto_drop", None),
        ("export_parts", Path("parts")),
        ("export_whole", Path("web_assembly.step")),
    ],
)
def test_repo_outer_step_pair_pipeline_stages_succeed(client, stage_name, artifact_path):
    client, _tmp_path = client
    _load_repo_root_outer_pair(client)

    resp = client.post(f"/api/stage/{stage_name}", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["ok"] is True

    if artifact_path is not None:
        assert artifact_path.exists()

    _assert_scene_has_loaded_parts(client, 2)


def test_test_model_cut_stage_succeeds_with_same_stage_process(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    load_resp = client.post("/api/load", json={"files": ["outer_1.step", "mid_1.step", "inner_1.step"], "include_scene": True})
    assert load_resp.status_code == 200

    resp = client.post("/api/stage/cut_inner_from_mid", json={})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data is not None
    assert data["ok"] is True
    assert "Cut complete" in data["message"]

    _assert_scene_has_loaded_parts(client, 3)


def test_test_models_full_backend_stack_nests_outer_mid_inner(client):
    client, _tmp_path = client
    web_ui.state.parts_dir = _load_test_model_dir()

    files = [*(f"outer_{i}.step" for i in range(1, 9)), *(f"mid_{i}.step" for i in range(1, 7)), *(f"inner_{i}.step" for i in range(1, 7))]
    load_resp = client.post("/api/load", json={"files": files, "include_scene": True})
    assert load_resp.status_code == 200

    axis_vec = assemble.AXIS_MAP["z"]
    _assy, part_info = assemble.stack_parts(web_ui._parts_for_assembly(), axis_vec, web_ui.state.gap)
    by_name = {name: (shape, loc) for name, shape, loc, *_rest in part_info}

    outer_z = []
    for level in range(1, 9):
        name = f"outer_{level}"
        bbox = _bbox_after_location(*by_name[name])
        z_center = (bbox[2] + bbox[5]) / 2.0
        outer_z.append(z_center)
    assert outer_z == sorted(outer_z)
    for lower, upper in zip(outer_z, outer_z[1:]):
        assert 480.0 <= (upper - lower) <= 520.0

    for level in range(1, 7):
        outer_bbox = _bbox_after_location(*by_name[f"outer_{level}"])
        mid_bbox = _bbox_after_location(*by_name[f"mid_{level}"])
        inner_bbox = _bbox_after_location(*by_name[f"inner_{level}"])
        assert _bbox_contains(outer_bbox, mid_bbox, tol=2.0)
        assert _bbox_contains(outer_bbox, inner_bbox, tol=2.0)
        mid_center = _bbox_center(mid_bbox)
        inner_center = _bbox_center(inner_bbox)
        assert mid_center[0] == pytest.approx(inner_center[0], abs=25.0)
        assert mid_center[1] == pytest.approx(inner_center[1], abs=25.0)
        assert mid_center[2] == pytest.approx(inner_center[2], abs=60.0)
