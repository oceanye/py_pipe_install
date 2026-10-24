"""Synthetic geometry regression; this does not validate physical cameras."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.elevation_auto import analyze_elevation_auto_groups, cylinder_depth_grid
from pipe_twin.elevation_dataset import create_elevation_dataset, load_elevation_dataset, elevation_history_compatible
from pipe_twin.stereo_analyzer import _calibration_from_manifest, analyze_stereo_capture
from pipe_twin.capture_gui import catalog_from_model
from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.elevation_registration import register_elevation
from test_elevation_dataset import _calibration


def _scene():
    calibration = _calibration_from_manifest(_calibration())
    rotation, _ = cv2.Rodrigues(np.array([.12, 1.25, .08]))
    translation = np.array([0., -10., 650.])
    specs = []
    for i, (x, y, diameter) in enumerate([(0, -100, 24), (60, -20, 36), (-15, 65, 50), (75, 130, 64)]):
        specs.append({"pipe_id": f"P{i+1}", "centerline_world_mm": [[x,y,-300], [x,y,300]],
                      "nominal_diameter_mm": diameter, "color_srgb": ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"][i]})
    return calibration, specs, rotation, translation


def _render(calibration, specs, rotation, translation, installed=(0,1,2), *, seed=1, occluded=False, texture=False):
    """Independent ray/cylinder quadratic, with metric truth for both eyes."""
    images, depths = {}, {}
    rng = np.random.default_rng(seed)
    axis = rotation[:, 2]
    for role in ("left", "right"):
        camera = getattr(calibration, role)
        yy, xx = np.indices((camera.height, camera.width))
        k = camera.rectified_intrinsic
        rays = np.stack([(xx-k[0,2])/k[0,0], (yy-k[1,2])/k[1,1], np.ones(xx.shape)], axis=-1)
        origin = np.array([calibration.baseline_mm if role == "right" else 0., 0., 0.])
        z = np.full(xx.shape, 1300.)
        image = np.full((*xx.shape,3), 80, np.uint8)
        if texture:
            bx, by = (origin + rays * z[...,None])[..., :2].transpose(2,0,1)
            pattern = (30*np.sin(bx*1.7) * np.sin(by*1.13) + 20*np.sin(bx*.39+by*.62))
            image[:] = np.clip(100+pattern[...,None], 0, 255)
        for index in installed:
            spec = specs[index]
            center = rotation @ np.asarray(spec["centerline_world_mm"]).mean(axis=0)+translation
            delta = origin-center
            u = rays-(rays@axis)[...,None]*axis
            v = delta-(delta@axis)*axis
            a, b = np.sum(u*u, axis=-1), 2*(u@v)
            c = v@v-(spec["nominal_diameter_mm"]/2)**2
            disc = b*b-4*a*c
            t = np.full(xx.shape, np.inf)
            hit = (disc>0) & (a>1e-12)
            t[hit] = (-b[hit]-np.sqrt(disc[hit]))/(2*a[hit])
            points = origin+rays*np.where(hit,t,0)[...,None]
            station = (points-center)@axis
            hit &= (t>0) & (t<z) & (np.abs(station)<190)
            z[hit] = t[hit]
            rgb = np.array([int(spec["color_srgb"][i:i+2],16) for i in (1,3,5)])
            if texture:
                local = (points-center) @ rotation
                theta = np.arctan2(local[...,1],local[...,0])
                pigment = .76 + .12*np.sin(station*2.1)*np.sin(theta*31) + .10*np.sin(station*.43+theta*7)
                image[hit] = np.clip(pigment[hit,None]*rgb[::-1],0,255)
            else:
                image[hit] = rgb[::-1]
        if occluded:
            # A planar foreground slab masks only the expected fourth pipe.
            mask = yy > 306
            image[mask] = 100
            z[mask] = 400
        if not texture:
            z += rng.normal(0, .12, z.shape)
        # Independent capture content, not a changed timestamp or file wrapper.
        image[0,0] = [seed, seed+1, seed+2]
        images[role], depths[role] = image, z.astype(np.float32)
    images["depth"] = SimpleNamespace(left_depth_mm=depths["left"], right_depth_mm=depths["right"],
        left_valid=np.isfinite(depths["left"]), right_valid=np.isfinite(depths["right"]), audit={"status":"VALID", "source":"ANALYTIC_TEST_TRUTH"})
    return images


def _groups(n=1, *, installed=(0,1,2), occluded=False):
    calibration, specs, rotation, translation = _scene()
    groups=[]
    for i in range(n):
        groups.append(dict(_render(calibration,specs,rotation,translation,installed,seed=i+1,occluded=occluded),
            capture_id=f"CAP-{i}", captured_at=f"2026-09-12T10:00:{i*2:02d}+08:00", pair_healthy=True))
    return calibration,specs,groups


def test_infinite_cylinder_depth_is_camera_z_and_does_not_mutate_axis():
    calibration, _, _, _ = _scene()
    axis = np.array([3.,0.,0.])
    left, _, _ = cylinder_depth_grid(calibration.left,np.array([0.,0.,650.]),axis,25)
    right, _, _ = cylinder_depth_grid(calibration.right,np.array([0.,0.,650.]),axis,25,baseline_offset=60)
    assert left[240,320] == pytest.approx(625)
    assert right[240,320] == pytest.approx(625)
    assert np.array_equal(axis,[3,0,0])
    # For a vertical cylinder, baseline changes the ray hit as expected.
    right, _, _ = cylinder_depth_grid(calibration.right,np.array([60.,0.,650.]),np.array([0.,1.,0.]),25,baseline_offset=60)
    assert right[240,320] == pytest.approx(625)
    with pytest.raises(ValueError):
        cylinder_depth_grid(calibration.left,np.array([0.,0.,650.]),np.zeros(3),25)


def test_oblique_local_surfaces_register_and_locate_the_unseen_fourth_pipe():
    calibration,specs,groups=_groups()
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["registration"]["status"] == "MATCHED", result["registration"]
    assert result["counts"] == {"INSTALLED":3,"NOT_INSTALLED":0,"UNKNOWN":1}
    fourth=result["pipes"][3]["current_evidence"]
    assert fourth["free_space_candidate"]
    assert fourth["left"]["expected_depth_range_mm"][1]-fourth["left"]["expected_depth_range_mm"][0]>30
    assert not result["longitudinal_installation_segments_assessed"]
    measured = result["pipes"][0]["measurement"]
    assert measured["status"] == "MEASURED"
    assert measured["nominal_diameter_source"] == "MODEL_CATALOG"
    assert measured["measured_section"]["minimum_surface_depth_z_mm"] < measured["measured_section"]["centerline"]["depth_z_mm"]
    assert measured["model_predicted_section"] is not None
    assert result["pipes"][3]["measurement"]["status"] == "MODEL_PREDICTION_ONLY"
    assert result["pipes"][3]["measurement"]["measured_diameter_mm"] is None


def test_status_refresh_is_reported_with_capture_and_analysis_times():
    calibration, specs, groups = _groups()
    groups[0]["status_refresh"] = {
        "action": "STATUS_REFRESH", "requested_at": "2026-10-03T08:00:00+00:00"}
    result = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs)
    assert result["status_refresh"]["requested_at"] == "2026-10-03T08:00:00+00:00"
    audit = result["capture_audit"]["groups"][0]
    assert audit["status_refresh"]["action"] == "STATUS_REFRESH"
    assert audit["captured_at"] == groups[0]["captured_at"]


def test_known_three_pipes_do_not_turn_full_catalog_into_observations():
    calibration, specs, groups = _groups()
    result = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs,
                                           registration_settings={"present_pipe_count": 3})
    assert result["scene_inventory"]["present_pipe_count"] == 3
    assert result["scene_inventory"]["model_candidate_count"] == 4
    assert result["counts"] == {"INSTALLED": 3, "NOT_INSTALLED": 0, "UNKNOWN": 1}
    # Even good geometry conflicts with an explicit count of two; reject the
    # alignment instead of silently discarding a convenient observation.
    conflict = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs,
                                             registration_settings={"present_pipe_count": 2})
    assert conflict["registration"]["status"] == "SCENE_INVENTORY_CONFLICT"
    assert conflict["counts"]["UNKNOWN"] == 4
    assert all(p["measurement"]["measured_section"] is None for p in conflict["pipes"])


def test_center_distance_does_not_depend_on_unobservable_model_axis_translation():
    from pipe_twin.elevation_auto import evaluate_registered_capture
    calibration, specs, groups = _groups()
    result = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs)
    shifted = copy.deepcopy(result["registration"])
    shifted["translation_model_to_camera_mm"] = (np.array(shifted["translation_model_to_camera_mm"]) +
        10000 * np.array(shifted["camera_axis"])).tolist()
    evidence = evaluate_registered_capture(specs, result["local_surface"], shifted, groups[0],
                                           groups[0]["depth"], calibration, healthy=True)
    for row in result["pipes"]:
        original = row["measurement"]["model_predicted_section"]
        updated = evidence[row["pipe_id"]]["measurement"]["model_predicted_section"]
        assert updated["centerline"]["depth_z_mm"] == pytest.approx(original["centerline"]["depth_z_mm"], abs=1e-8)
        assert updated["centerline"]["range_mm"] == pytest.approx(original["centerline"]["range_mm"], abs=1e-8)
        if "center_distance_error_mm" in row["current_evidence"]:
            assert evidence[row["pipe_id"]]["center_distance_error_mm"] == pytest.approx(row["current_evidence"]["center_distance_error_mm"], abs=1e-8)


@pytest.mark.parametrize("count", [0, -1, True, 3.5, "3", 129])
def test_invalid_scene_count_rejected(count):
    from pipe_twin.elevation_dataset import normalize_registration_settings
    with pytest.raises(ValueError, match="现场实际管数"):
        normalize_registration_settings({"present_pipe_count": count})


def test_scene_count_survives_package_roundtrip(tmp_path):
    path, _ = _package(tmp_path, registration_settings={"present_pipe_count": 3})
    assert load_elevation_dataset(path)["registration_settings"]["present_pipe_count"] == 3


def test_two_independent_registered_free_space_captures_establish_absence():
    calibration,specs,groups=_groups(2)
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["counts"] == {"INSTALLED":3,"NOT_INSTALLED":1,"UNKNOWN":0}, result["registration"]
    assert result["pipes"][3]["independent_free_space_captures"]==["CAP-0","CAP-1"]


def test_duplicate_pixels_and_foreground_occlusion_do_not_prove_absence():
    calibration,specs,groups=_groups(2)
    for role in ("left","right"):
        groups[1][role]=groups[0][role].copy()
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["pipes"][3]["installation_state"]=="UNKNOWN"
    calibration,specs,groups=_groups(2,occluded=True)
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["pipes"][3]["installation_state"]=="UNKNOWN"
    assert "FOREGROUND_OCCLUSION" in result["pipes"][3]["reason_codes"]


def test_no_reconstruction_or_unhealthy_capture_fails_to_unknown():
    calibration,specs,groups=_groups(installed=())
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["counts"]["UNKNOWN"]==4
    calibration,specs,groups=_groups()
    groups[0]["pair_healthy"]=False
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["counts"]["UNKNOWN"]==4
    assert all(p["measurement"]["measured_section"] is None for p in result["pipes"])
    assert all(p["measurement"]["model_predicted_section"] is None for p in result["pipes"])


def test_camera_move_does_not_reuse_old_free_space():
    calibration,specs,groups=_groups(2)
    _,_,rotation,translation=_scene()
    groups[1].update(_render(calibration,specs,rotation,translation+[0,12,0],seed=2))
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["pipes"][3]["installation_state"]=="UNKNOWN"


def test_display_colors_do_not_define_physical_pipe_identity():
    calibration,specs,groups=_groups()
    for spec in specs:
        spec["color_srgb"]="#B0B0B0"
    result=analyze_elevation_auto_groups(groups,calibration=calibration,pipe_specs=specs)
    assert result["counts"]=={"INSTALLED":3,"NOT_INSTALLED":0,"UNKNOWN":1}
    assert all("depth_connected_surface" in obs["segmentation_sources"] for obs in result["local_surface"]["observations"])


def _package(tmp_path, **kwargs):
    model=Path(__file__).resolve().parents[1]/"test_model"/"管道布置.stl"
    specs,_=catalog_from_model(model,stl_unit="millimeter")
    left=np.zeros((480,640,3),np.uint8)
    right=np.full_like(left,16)
    paths={r:tmp_path/f"{r}.png" for r in ("left","right")}
    cv2.imencode(".png",left)[1].tofile(str(paths["left"]))
    cv2.imencode(".png",right)[1].tofile(str(paths["right"]))
    settings=dict(output_root=tmp_path/"packages",calibration=_calibration(),left_path=paths["left"],right_path=paths["right"],
        left_time="2026-09-12T10:00:00+08:00",right_time="2026-09-12T10:00:00.001+08:00",pipe_specs=specs,
        pair_confirmed=True,model_path=model,stl_unit="millimeter",mode="elevation_auto")
    settings.update(kwargs)
    return create_elevation_dataset(**settings),settings


def test_auto_package_needs_no_regions_and_restores_stl_axis_and_anchor_hashes(tmp_path):
    path,settings=_package(tmp_path,registration_settings={"axis_world":[0,0,-2],"anchors":{"OBS-0001":"P001"}})
    loaded=load_elevation_dataset(path)
    assert loaded["mode"]=="elevation_auto"
    assert loaded["registration_settings"]["axis_world"]==[0,0,-1]
    assert not any("left_region_px" in p for p in loaded["pipe_specs"])
    payload=loaded["manifest"]["analysis"]["elevation_auto"]
    assert payload["anchor_pair_sha256"]["left"]==loaded["manifest"]["capture"]["capture_groups"][-1]["views"]["left"]["sha256"]
    common=dict(model_path=settings["model_path"],stl_unit="millimeter",mode="elevation_auto")
    assert elevation_history_compatible(path,settings["calibration"],settings["pipe_specs"],registration_settings=loaded["registration_settings"],**common)
    assert not elevation_history_compatible(path,settings["calibration"],settings["pipe_specs"],registration_settings={"axis_world":[0,0,1]},**common)


def test_camera_side_roundtrip_and_history_invalidation(tmp_path):
    path, settings = _package(tmp_path, registration_settings={"camera_side_world": [0, -3, 0]})
    loaded = load_elevation_dataset(path)
    assert loaded["registration_settings"]["camera_side_world"] == [0, -1, 0]
    common = dict(model_path=settings["model_path"], stl_unit="millimeter", mode="elevation_auto")
    for side, compatible in [([0, -1, 0], True), ([0, 1, 0], False), (None, False)]:
        assert elevation_history_compatible(path, settings["calibration"], settings["pipe_specs"],
            registration_settings={"camera_side_world": side}, **common) == compatible


def test_wrong_camera_side_keeps_all_installation_states_unknown():
    calibration, specs, groups = _groups()
    _, _, rotation, _ = _scene()
    correct_side = (rotation.T @ [0., 0., -1.]).tolist()
    correct = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs,
        registration_settings={"camera_side_world": correct_side})
    assert correct["counts"] == {"INSTALLED": 3, "NOT_INSTALLED": 0, "UNKNOWN": 1}
    wrong = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs,
        registration_settings={"camera_side_world": (-np.asarray(correct_side)).tolist()})
    assert wrong["counts"] == {"INSTALLED": 0, "NOT_INSTALLED": 0, "UNKNOWN": 4}
    assert "CAMERA_SIDE_CONFLICT" in wrong["registration"]["reason_codes"]
    assert all("CAMERA_SIDE_CONFLICT" in p["reason_codes"] for p in wrong["pipes"])


def test_real_dxf_layout_is_an_automatic_model_without_stl_or_mesh(tmp_path):
    model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.dxf"
    specs, skipped = catalog_from_dxf(model)
    assert len(specs) == 12 and len(skipped) == 4
    left = np.zeros((480, 640, 3), np.uint8)
    right = np.full_like(left, 16)
    left_path, right_path = tmp_path / "left.png", tmp_path / "right.png"
    cv2.imwrite(str(left_path), left); cv2.imwrite(str(right_path), right)
    path = create_elevation_dataset(
        output_root=tmp_path / "packages", calibration=_calibration(),
        left_path=left_path, right_path=right_path,
        left_time="2026-09-12T10:00:00+08:00", right_time="2026-09-12T10:00:00.001+08:00",
        pipe_specs=specs, pair_confirmed=True, model_path=model,
        stl_unit="millimeter", mode="elevation_auto",
        registration_settings={"axis_world": [0, 0, 1]},
    )
    loaded = load_elevation_dataset(path)
    assert loaded["manifest"]["model"]["format"] == "dxf"
    assert len(loaded["pipe_specs"]) == 12
    assert {round(p["nominal_diameter_mm"]): p["color_srgb"] for p in loaded["pipe_specs"]} == {26: "#FFFFFF", 41: "#FF0000", 51: "#0000FF"}


def test_three_visible_dxf_pipes_are_identified_by_diameter_and_distance_before_color():
    model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.dxf"
    specs, _ = catalog_from_dxf(model)
    rotation, _ = cv2.Rodrigues(np.array([.12, 1.25, .08]))
    translation = np.array([0.0, -10.0, 2650.0])
    observations = []
    for index in (0, 1, 2):
        spec = specs[index]
        center = rotation @ np.mean(np.asarray(spec["centerline_world_mm"], dtype=float), axis=0) + translation
        observations.append({
            "observation_id": f"OBS-{index}",
            "center_camera_mm": center.tolist(),
            "axis_camera": (rotation @ np.array([0.0, 0.0, 1.0])).tolist(),
            "diameter_mm": spec["nominal_diameter_mm"],
            # Deliberately different display colour: color is auxiliary.
            "color_srgb": "#808080",
            "color_identity_validated": False,
        })
    result = register_elevation(specs, observations, axis_world=[0, 0, 1])
    assert result["status"] == "MATCHED"
    assert {row["pipe_id"] for row in result["matches"]} == {"P001", "P002", "P003"}
    assert result["rms_mm"] < 1e-6


def test_tampered_auto_geometry_and_omitted_stl_components_are_rejected(tmp_path):
    path,_=_package(tmp_path)
    manifest=json.loads(path.read_text(encoding="utf-8"))
    manifest["analysis"]["elevation_auto"]["pipes"][0]["centerline_world_mm"][0][0]+=5
    path.write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError,match="中心线"):
        load_elevation_dataset(path)
    manifest["analysis"]["elevation_auto"]["pipes"].pop(0)
    path.write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError,match="组件"):
        load_elevation_dataset(path)


def test_cross_section_direction_is_rejected_before_capture_processing(tmp_path):
    with pytest.raises(ValueError,match="方向须沿管长"):
        _package(tmp_path,registration_settings={"axis_world":[1,0,0]})


def test_manifest_dispatch_actual_sgbm_and_report_exports_are_closed_on_blank_images(tmp_path):
    path,_=_package(tmp_path)
    report=tmp_path/"report.json"
    result=analyze_stereo_capture(path,report_output_path=report,evidence_dir=tmp_path/"evidence")
    assert result["mode"]=="elevation_auto"
    assert result["counts"]=={"INSTALLED":0,"NOT_INSTALLED":0,"UNKNOWN":12}
    assert json.loads(report.read_text(encoding="utf-8"))["registration"]["status"]=="INSUFFICIENT_OBSERVATIONS"
    assert len(result["evidence_files"])==3
    assert all(Path(name).is_file() for name in result["evidence_files"])
    assert "element vertex 0" in (tmp_path/"evidence"/"local_surface.ply").read_text()
    with pytest.raises(ValueError):
        analyze_stereo_capture(path,report_output_path=path)


def test_anchor_binding_cannot_be_replayed_on_other_photos(tmp_path):
    path,_=_package(tmp_path,registration_settings={"anchors":{"OBS-0001":"P001"}})
    manifest=json.loads(path.read_text(encoding="utf-8"))
    manifest["analysis"]["elevation_auto"]["anchor_pair_sha256"]["left"]="0"*64
    path.write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError,match="照片哈希"):
        analyze_stereo_capture(path)
