import copy

import pytest

from pipe_twin.elevation_dataset import normalize_registration_settings
from pipe_twin.installation_assist_config import normalize_installation_assist_settings


def test_qualitative_purchase_hints_are_portable_separate_from_measurement_settings():
    source = {'camera_side_world': [0, 1, 0], 'installation_assist': {
        'appearance_diameter_hints': {'blue': 50, 'RED': 40, 'WHITE': 25},
        'hint_source': 'USER_STATED_PURCHASE_DIAMETERS'}}
    before = copy.deepcopy(source)
    normalized = normalize_registration_settings(source)
    assert normalized['installation_assist']['appearance_diameter_hints'] == {'BLUE': 50., 'RED': 40., 'WHITE': 25.}
    assert normalized['camera_side_world'] == [0., 1., 0.]
    assert normalized['anchors'] == {} and 'known_diameter_mm' not in normalized
    assert normalize_registration_settings(normalized) == normalized
    assert source == before
    assert 'installation_assist' not in normalize_registration_settings({})


@pytest.mark.parametrize('value', [True, -1, 0, float('nan'), float('inf'), '50'])
def test_invalid_purchase_hint_cannot_be_loaded(value):
    with pytest.raises(ValueError):
        normalize_registration_settings({'installation_assist': {'appearance_diameter_hints': {'BLUE': value}}})


def test_invalid_or_duplicate_hint_keys_are_not_silently_reused():
    with pytest.raises(ValueError):
        normalize_installation_assist_settings({'appearance_diameter_hints': {'blue': 40, 'BLUE': 50}})
    with pytest.raises(ValueError):
        normalize_installation_assist_settings({'installation_state': 'INSTALLED'})
    with pytest.raises(ValueError):
        normalize_installation_assist_settings({'appearance_diameter_hints': {'NO_SUCH_COLOR': 50}})


def test_portable_hints_do_not_invent_installations_from_blank_photos(tmp_path):
    from test_elevation_auto import _package
    from pipe_twin.elevation_dataset import load_elevation_dataset
    from pipe_twin.stereo_analyzer import analyze_stereo_capture
    hints = {'appearance_diameter_hints': {'BLUE': 50., 'RED': 40., 'WHITE': 25.},
             'hint_source': 'USER_STATED_PURCHASE_DIAMETERS'}
    path, _ = _package(tmp_path, registration_settings={'installation_assist': hints})
    before = path.read_bytes()
    loaded = load_elevation_dataset(path)
    assert loaded['registration_settings']['installation_assist'] == hints
    report = analyze_stereo_capture(path)
    assist = report['installation_assist']
    assert assist['advisory_only'] and not assist['accepted_measurement']
    assert assist['counts'] == {'LIKELY': 0, 'AMBIGUOUS': 0, 'NO_OBSERVATION': 12}
    assert report['counts'] == {'INSTALLED': 0, 'NOT_INSTALLED': 0, 'UNKNOWN': 12}
    source = loaded['manifest']['capture']['capture_groups'][-1]
    assert assist['source']['photo_sha256'] == {role: source['views'][role]['sha256'] for role in ('left', 'right')}
    assert path.read_bytes() == before
