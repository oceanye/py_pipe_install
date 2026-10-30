"""Inspect the frozen field result and verify direction saving in an isolated GUI."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.ground_visualization import tilt_angles

SCENE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261030-161444-3ab7011e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((SCENE / 'manifest.json').read_text(encoding='utf8'))
    report = json.loads((SCENE / 'report.json').read_text(encoding='utf8'))
    section = report['section_evaluations'][0]
    ground = report['ground_check']
    protected = [SCENE / 'manifest.json', SCENE / 'report.json',
                 SCENE / 'photos/left.png', SCENE / 'photos/right.png',
                 ROOT / 'outputs/elevation_workbench/last_scene.json',
                 ROOT / 'outputs/measurement_workbench/calibration_current.json']
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    evidence = {
        'dataset_id': manifest['dataset_id'], 'generated_at': report['generated_at'],
        'sources_sha256': before,
        'calibration_id': manifest['stereo_calibration']['calibration_id'],
        'calibration_validated': manifest['stereo_calibration']['validated'],
        'scale_verified_independently': manifest['stereo_calibration']['quality_audit']['scale_verified_independently'],
        'capture_time': manifest['capture']['capture_groups'][0]['views']['left']['captured_at'],
        'temporal_sections': section['temporal_sections'],
        'ground': {key: ground[key] for key in ['status', 'accepted_measurement', 'selection',
                    'valid_depth_fraction', 'left_plane', 'correction_status', 'corrections']},
        'ground_angles_degrees_optical_axis_to_plane_and_roll': tilt_angles(ground),
        'ground_trial_comparison': section['ground_trial_comparison'],
        'saved_camera_side_world': manifest['analysis']['elevation_auto']['registration'].get('camera_side_world'),
        'report_camera_view': section['registration']['camera_view'],
        'registration_status': section['registration']['status'],
        'installation_counts': report['counts'],
        'position_counts': report['position_assessment']['counts'],
        'candidate_results': [{k: row[k] for k in ['label', 'candidate_id', 'color_srgb', 'solved',
                              'diameter_mm', 'pipe_id', 'reason_codes']} for row in report['solver_status']],
        'matching_settings': manifest['analysis']['matching'],
        'user_accepted_nominal_diameter_difference_mm': 1.0,
    }
    root = tk.Tk()
    root.withdraw()
    messages = Mock()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=messages, filedialog=Mock())
    out = ROOT / 'outputs/flow-inspection-20261030/gui-test'
    dialog = ElevationCaptureDialog(app, output_root=out, restore=False)
    dialog.window.withdraw()
    try:
        dialog.load_session(SCENE / 'manifest.json')
        dialog.open_model_viewer()
        viewer = dialog.model_viewer
        viewer.window.withdraw()
        assert dialog.registration_settings.get('camera_side_world') is None
        assert '未指定' in viewer.camera_side_status()
        # This direction is a software test fixture, NOT the user's field direction.
        viewer.view_mode.set('从 +Y 侧朝 -Y 看')
        viewer.preview_side()
        assert '仅预览' in viewer.info.get()
        assert dialog.registration_settings.get('camera_side_world') is None
        viewer.apply_camera_side()
        assert dialog.registration_settings['camera_side_world'] == [0., 1., 0.]
        assert dialog.camera_side_label.get() == viewer.view_mode.get()
        caption = viewer.section_canvas.itemcget(viewer.section_canvas.find_withtag('camera_side_status')[0], 'text')
        assert '已应用' in caption and '待应用' not in caption
        assert viewer.section_canvas.find_withtag('camera_direction')
        dialog.confirmed.set(True)
        dialog.history.set(False)
        dialog.submit(False)
        deadline = time.monotonic() + 45
        while dialog.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert not dialog.busy and not messages.showerror.called, messages.mock_calls
        saved = dialog.last_manifest
        assert saved != SCENE / 'manifest.json' and out.resolve() in saved.resolve().parents
        test_manifest = json.loads(saved.read_text(encoding='utf8'))
        assert test_manifest['analysis']['elevation_auto']['registration']['camera_side_world'] == [0., 1., 0.]
        dialog.load_session(saved)
        assert dialog.registration_settings['camera_side_world'] == [0., 1., 0.]
        evidence['gui_verification'] = {
            'scope': 'ISOLATED_TEST_DIRECTION_NOT_USER_FIELD_SELECTION',
            'preview_does_not_change_matching': True, 'applied_caption': caption,
            'main_control_and_section_arrow_linked': True,
            'saved_and_reopened_direction': dialog.registration_settings['camera_side_world'],
            'test_manifest': str(saved.relative_to(ROOT)),
        }
    finally:
        dialog.close()
        root.destroy()
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    evidence['source_scene_photos_calibration_and_default_pointer_unchanged'] = True
    target = Path(__file__).with_name('evidence.json')
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps({'evidence': str(target), 'gui': evidence['gui_verification']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
