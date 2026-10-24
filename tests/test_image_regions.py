"""Image proposals may guide measured geometry but never confer identity."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from pipe_twin.image_regions import filter_disparity, propose_regions
from pipe_twin.local_surface import extract_local_pipes
from pipe_twin.recognition import recognize_local_pipes
from pipe_twin.elevation_auto import analyze_elevation_auto_groups, analyze_elevation_auto_zones
from pipe_twin.elevation_gui import recognition_summary, relative_layout_summary_lines
from pipe_twin.elevation_registration import register_elevation
from test_elevation_auto import _groups
from test_elevation_registration import _specs, _observations


def test_image_proposals_split_parallel_paint_and_ignore_catalogue():
    im = np.full((320,640,3),60,np.uint8)
    cv2.rectangle(im,(40,40),(590,90),(30,30,230),-1)
    cv2.rectangle(im,(40,125),(590,175),(30,30,230),-1)
    # Thin transverse connector must not turn the two pipes into one region.
    cv2.rectangle(im,(300,75),(312,140),(30,30,230),-1)
    cv2.rectangle(im,(40,230),(590,280),(220,110,10),-1)
    cv2.line(im,(70,253),(560,253),(245,245,245),4)
    before = im.copy()
    regions,audit = propose_regions(im)
    assert not audit['truncated']
    assert len(regions)==3
    assert [r['region_px'][1] for r in regions] == pytest.approx([40,125,230],abs=3)
    assert np.array_equal(im,before)
    assert all('pipe_id' not in r and 'diameter_mm' not in r for r in regions)


def test_background_majority_direction_does_not_erase_visible_pipes():
    """Carpet/stand fragments can outvote pipes differently in each eye."""
    image = np.full((480, 800, 3), 60, np.uint8)
    cv2.rectangle(image, (40, 40), (750, 90), (220, 110, 10), -1)
    cv2.rectangle(image, (40, 130), (750, 175), (30, 30, 230), -1)
    for x in range(20, 780, 60):
        cv2.rectangle(image, (x, 230), (x+12, 440), (200, 200, 200), -1)
    regions, audit = propose_regions(image)
    paints = [r for r in regions if r['appearance'] != 'neutral']
    assert len(paints) == 2
    assert all(abs(r['axis_image'][0]) > .99 for r in paints)
    assert len(regions) <= 24
    assert not audit['truncated']


def test_hue_boundary_does_not_fragment_one_pipe_or_bridge_a_dark_gap():
    """A change of blue shade across a bin boundary is not a pipe boundary."""
    hsv = np.zeros((260, 640, 3), np.uint8)
    hsv[:, :, 2] = 45
    for y0 in (30, 150):
        for x in range(40, 600):
            split = y0 + 40 + round(25*np.sin(x/70))
            hsv[y0:split, x] = (102, 230, 210)
            hsv[split:y0+80, x] = (108, 230, 210)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    before = image.copy()
    regions, audit = propose_regions(image)
    assert len(regions) == 2
    assert [r['width_px'] for r in regions] == pytest.approx([79, 79], abs=2)
    assert all(r['length_px'] > 550 for r in regions)
    assert all(r['region_px'][3] < 85 for r in regions)
    assert not audit['truncated']
    np.testing.assert_array_equal(image, before)


def test_hue_overlap_preserves_distinct_touching_colours():
    hsv = np.zeros((200, 600, 3), np.uint8)
    hsv[:, :, 2] = 45
    hsv[30:90, 40:560] = (60, 230, 210)
    hsv[90:150, 40:560] = (120, 230, 210)
    regions, _ = propose_regions(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
    assert len(regions) == 2
    assert [r['width_px'] for r in regions] == pytest.approx([59, 59], abs=2)


def test_clipped_width_is_distinguished_from_clipped_pipe_ends():
    image = np.full((320, 640, 3), 60, np.uint8)
    cv2.rectangle(image, (40, 0), (590, 30), (30, 30, 230), -1)
    cv2.rectangle(image, (0, 125), (639, 175), (220, 110, 10), -1)
    regions, _ = propose_regions(image)
    red = next(r for r in regions if r['appearance'] == 'hue-0')
    blue = next(r for r in regions if r['appearance'] == 'hue-120')
    assert red['transverse_boundary_contacts'] == ['top']
    assert not red['axially_clipped']
    assert blue['boundary_contacts'] == ['left', 'right']
    assert blue['axially_clipped']
    assert not blue['transverse_boundary_contacts']


def test_gui_explains_missing_depth_and_clipped_candidate_without_claiming_measurement():
    from pipe_twin.elevation_gui import recognition_diagnostic_lines
    report = {'local_surface': {'image_candidates': [{
        'candidate_id': 'IMG-left-001', 'role': 'left', 'region_px': [10, 0, 300, 30],
        'axis_to_epipolar_deg': 2., 'valid_depth_fraction': 0.,
        'correspondence_diagnostic': {'surface_texture': {'sample_count': 512, 'low_texture_fraction': .95},
                                    'warning_codes': ['CANDIDATE_WIDTH_CLIPPED', 'LOW_SURFACE_HORIZONTAL_TEXTURE']}}]}}
    text = '\n'.join(recognition_diagnostic_lines(report))
    assert '95.0%' in text and '包含没有深度的区域' in text
    assert '保留完整管宽' in text and '2.0°' in text
    assert '图像唯一匹配：未计算' in text and '合格三维管段 0' in text


def test_gui_distinguishes_static_scene_from_inconsistent_matching():
    from pipe_twin.elevation_gui import recognition_diagnostic_lines
    report = {'temporal_fusion': {'status': 'VALID', 'accepted_frames': 20, 'requested_frames': 20,
        'frames': [{'motion': {r: {'passed': True} for r in ('left', 'right')}}],
        'left': {'required_support_frames': 10, 'majority_cluster_recovered_pixels': 25,
                 'insufficient_observation_pixels': 40, 'inconsistent_observation_pixels': 10,
                 'pre_consistency_valid_fraction': .30, 'post_consistency_valid_fraction': .25}}}
    text = '\n'.join(recognition_diagnostic_lines(report))
    assert '本序列通过静止检查' in text
    assert '不表示设备或管道发生移动' in text
    assert '重新找回 25 个像素' in text and '30.0% → 左右复核后 25.0%' in text
    assert '包含背景，不代表管面合格点数' in text
    report['temporal_fusion']['frames'][0]['motion']['left']['passed'] = False
    assert '本序列通过静止检查' not in '\n'.join(recognition_diagnostic_lines(report))


def test_disparity_filter_removes_outliers_and_never_fills_holes():
    rng=np.random.default_rng(55)
    yy,xx=np.indices((80,300))
    true_disparity=90+.004*xx+.001*yy
    depth=60000/true_disparity
    outliers=rng.random(depth.shape)<.25
    depth[outliers]=rng.uniform(1200,3000,outliers.sum())
    support=rng.random(depth.shape)>.5
    before=depth.copy()
    kept,audit=filter_disparity(depth,support,SimpleNamespace(fx=1000),60,[1,0])
    assert audit['reason'] is None
    assert not (kept & ~support).any()
    assert not (kept & outliers).any()
    assert kept.sum() > .7*support.sum()
    assert np.array_equal(before,depth)


def test_disparity_filter_rejects_incoherent_and_short_support():
    rng=np.random.default_rng(55)
    depth=rng.uniform(700,4500,(80,300))
    kept,audit=filter_disparity(depth,np.ones(depth.shape,bool),SimpleNamespace(fx=1000),60,[1,0])
    assert not kept.any()
    assert audit['reason']=='NO_DOMINANT_DISPARITY_SUPPORT'
    kept,audit=filter_disparity(depth,np.zeros(depth.shape,bool),SimpleNamespace(fx=1000),60,[1,0])
    assert not kept.any() and audit['reason']=='INSUFFICIENT_VALID_DEPTH'


def test_disparity_filter_rejects_two_competing_depth_layers():
    rng=np.random.default_rng(42)
    depth=np.where(rng.random((80,300))<.5,750.,1800.)
    kept,audit=filter_disparity(depth,np.ones(depth.shape,bool),SimpleNamespace(fx=1000),60,[1,0])
    assert not kept.any()
    assert audit['reason']=='NO_DOMINANT_DISPARITY_SUPPORT'


def _sparse_noisy_scene():
    cal,specs,groups=_groups()
    rng=np.random.default_rng(93)
    for role in ('left','right'):
        valid=getattr(groups[0]['depth'],role+'_valid')
        z=getattr(groups[0]['depth'],role+'_depth_mm')
        valid[:]=rng.random(valid.shape)>.65
        bad=(rng.random(z.shape)<.2)&valid
        z[bad]+=rng.uniform(200,2000,bad.sum())
    return cal,specs,groups


def test_auto_recovers_sparse_cylinders_without_changing_cad_or_depth():
    cal,specs,groups=_sparse_noisy_scene()
    g=groups[0]
    before=copy.deepcopy(g['depth'])
    pure=extract_local_pipes(g['left'],g['right'],g['depth'],cal,specs,config={'geometry_only':True})
    assert not pure['observations']
    # Instrument the real call, proving fallback is a different algorithm.
    with patch('pipe_twin.local_surface.extract_local_pipes',wraps=extract_local_pipes) as calls:
        result=analyze_elevation_auto_groups(groups,calibration=cal,pipe_specs=specs,
                                             matching_settings={'color_filter_enabled':False})
    assert [c.kwargs['config'].get('image_guided',False) for c in calls.call_args_list]==[False,True]
    assert result['counts']=={'INSTALLED':3,'NOT_INSTALLED':0,'UNKNOWN':1}
    surface=result['local_surface']
    assert surface['audit']['recognizer_backend_used']=='image_guided'
    assert sorted(o['diameter_mm'] for o in surface['observations'])==pytest.approx([24,36,50],abs=.3)
    assert all(o['observation_basis']=='IMAGE_GUIDED_STEREO_CYLINDER' for o in surface['observations'])
    for role in ('left','right'):
        assert np.array_equal(getattr(before,role+'_depth_mm'),getattr(g['depth'],role+'_depth_mm'))
        assert np.array_equal(getattr(before,role+'_valid'),getattr(g['depth'],role+'_valid'))
    layout=result['registration']['relative_layout']
    assert layout['max_pairwise_error_mm']<1
    assert not layout['camera_distance_used_as_model_target']
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('condition',['missing','random','flat'])
def test_paint_candidates_do_not_turn_bad_depth_into_installed(condition):
    cal,specs,groups=_groups()
    rng=np.random.default_rng(8)
    for role in ('left','right'):
        z=getattr(groups[0]['depth'],role+'_depth_mm')
        valid=getattr(groups[0]['depth'],role+'_valid')
        if condition=='missing':
            valid[:]=False
        elif condition=='flat':
            z[:]=900
        else:
            z[:]=rng.uniform(700,4000,z.shape)
    result=analyze_elevation_auto_groups(groups,calibration=cal,pipe_specs=specs,
        registration_settings={'local_observation_mode':'image_guided'},matching_settings={'color_filter_enabled':False})
    assert result['counts']=={'INSTALLED':0,'NOT_INSTALLED':0,'UNKNOWN':4}
    assert len(result['local_surface']['image_candidates'])>=6
    assert not result['local_surface']['observations']
    assert all(p['measurement']['measured_diameter_mm'] is None for p in result['pipes'])
    assert '尚无可靠' in '\n'.join(relative_layout_summary_lines(result))
    assert '合格三维管段 0' in recognition_summary(result)


def test_zone_candidate_coordinates_and_ids_are_global_and_serializable():
    cal,specs,groups=_groups()
    for role in ('left','right'):
        getattr(groups[0]['depth'],role+'_valid')[:]=False
    zone={'zone_id':'Z09','roi_rect_px':[120,100,390,240]}
    r=analyze_elevation_auto_zones(groups,calibration=cal,pipe_specs=specs,
        scope_settings={'scope':'zones','zones':[zone]},matching_settings={'color_filter_enabled':False})
    candidates=r['local_surface']['image_candidates']
    assert candidates
    assert candidates==r['capture_audit']['groups'][0]['image_candidates']
    for p in candidates:
        assert p['candidate_id'].startswith('Z09:')
        assert p['region_px'][1]>=100
        if p['role']=='left':
            assert p['region_px'][0]>=120
    assert all(p['installation_state_zh']=='不确定' for p in r['pipes'])
    json.dumps(r,allow_nan=False)


def test_relative_diameters_and_clearances_ignore_camera_translation_rotation():
    specs=_specs()
    observed=_observations(specs)
    observed[0]['diameter_mm']+=1
    before=register_elevation(specs,observed)
    rotation,_=cv2.Rodrigues(np.array([.27,-.36,.18]))
    for obs in observed:
        obs['center_camera_mm']=(rotation @ np.asarray(obs['center_camera_mm'])+[130,-230,1800]).tolist()
        obs['axis_camera']=(rotation @ np.asarray(obs['axis_camera'])).tolist()
    after=register_elevation(specs,observed)
    assert before['status']==after['status']=='MATCHED'
    for a,b in zip(before['relative_layout']['pairwise'],after['relative_layout']['pairwise']):
        assert b['observed_distance_mm']==pytest.approx(a['observed_distance_mm'],abs=1e-8)
        assert b['observed_surface_gap_mm']==pytest.approx(a['observed_surface_gap_mm'],abs=1e-8)
        if a['pipe_id_a']=='P1' or a['pipe_id_b']=='P1':
            assert a['surface_gap_error_mm']==pytest.approx(-.5,abs=1e-8)
    text='\n'.join(relative_layout_summary_lines({'registration':after}))
    assert '轴心距' in text and '表面净距' in text and '+1.00' in text


def test_partial_primary_surface_fallback_keeps_observations_without_audit_cycle():
    # A truncated primary path with useful partial geometry must not disappear
    # merely because the fallback has no fits; nested audits must serialize.
    first={'observations':[{'observation_id':'kept'}], 'audit':{'truncated':True}}
    second={'observations':[],'image_candidates':[], 'audit':{'truncated':False}}
    with patch('pipe_twin.local_surface.extract_local_pipes',side_effect=[first,second]):
        r=recognize_local_pipes('auto',None,None,None,None,[],matching_settings={'color_filter_enabled':False})
    assert r['observations']==[{'observation_id':'kept'}]
    assert r['audit']['truncated']
    json.dumps(r)


def test_legacy_strip_representative_points_are_not_cylinder_axes():
    specs=_specs()
    observations=_observations(specs)
    for o in observations:
        o['parallel_local']=True
    r=register_elevation(specs,observations)
    assert r['status']=='MATCHED'
    assert not r['relative_layout']['axis_centers_resolved']
    assert all(p['observed_surface_gap_mm'] is None for p in r['relative_layout']['pairwise'])
    assert '圆柱轴心未确认' in '\n'.join(relative_layout_summary_lines({'registration':r}))


def test_zone_relative_layout_displays_qualified_pairs():
    cal,specs,groups=_groups()
    result=analyze_elevation_auto_zones(groups,calibration=cal,pipe_specs=specs,
        scope_settings={'scope':'zones','zones':[{'zone_id':'Z09','roi_rect_px':[120,0,440,480]}]},
        registration_settings={'local_observation_mode':'image_guided'})
    reg=result['zone_registration_audit']['registrations'][0]
    assert reg['status']=='MATCHED'
    assert reg['zone_id']=='Z09'
    assert all(p['observation_id_a'].startswith('Z09:') and p['observation_id_b'].startswith('Z09:')
               for p in reg['relative_layout']['pairwise'])
    text='\n'.join(relative_layout_summary_lines(result))
    assert 'Z09' in text and '表面净距：' in text


@pytest.mark.parametrize('disjoint',[False,True])
def test_local_windows_require_common_clean_segment_without_loosening_fit(disjoint):
    cal,specs,groups = _groups(installed=(0,))
    group = groups[0]
    rng = np.random.default_rng(9)
    for role in ('left','right'):
        z = getattr(group['depth'],role+'_depth_mm')
        k = getattr(cal,role).rectified_intrinsic
        yy,xx = np.indices(z.shape)
        world_x = (xx-k[0,2])*z/k[0,0] + (cal.baseline_mm if role=='right' else 0)
        bad = (group[role][:,:,2]>150) & ((world_x>0) if disjoint and role=='right' else (world_x<0))
        z[bad] = rng.uniform(1400,3000,bad.sum())
    before = copy.deepcopy(group['depth'])
    args = (group['left'],group['right'],group['depth'],cal,specs)
    with patch('pipe_twin.image_regions.axial_windows',return_value=[]):
        whole = extract_local_pipes(*args,config={'image_guided':True})
    local = extract_local_pipes(*args,config={'image_guided':True})
    assert not whole['observations']
    assert len(local['observations'])==(0 if disjoint else 1)
    if not disjoint:
        observation = local['observations'][0]
        assert observation['diameter_mm']==pytest.approx(24,abs=.1)
        assert observation['left_right_center_difference_mm']<.1
    assert all(len(c['attempts'])<=4 for c in local['image_candidates'])
    assert all(c['attempts'][0]['fit_count']==0 for c in local['image_candidates'])
    for role in ('left','right'):
        np.testing.assert_array_equal(getattr(before,role+'_depth_mm'),getattr(group['depth'],role+'_depth_mm'))
    json.dumps(local,allow_nan=False)


def test_broad_highlight_joins_paint_without_joining_dark_interpipe_gap():
    image = np.full((250,500,3),45,np.uint8)
    cv2.rectangle(image,(30,25),(460,125),(220,90,10),-1)
    cv2.rectangle(image,(30,68),(460,82),(255,255,245),-1)
    cv2.rectangle(image,(30,155),(460,215),(220,90,10),-1)
    regions,_ = propose_regions(image)
    assert len(regions)==2
    assert regions[0]['width_px']==pytest.approx(100,abs=2)
    assert regions[1]['width_px']==pytest.approx(60,abs=2)


def test_crop_keeps_thick_pipe_proposal_instead_of_only_its_highlight():
    image = np.full((250,500,3),45,np.uint8)
    cv2.rectangle(image,(30,25),(460,125),(220,90,10),-1)
    cv2.rectangle(image,(30,68),(460,82),(255,255,245),-1)
    regions,_ = propose_regions(image[:,120:283])
    assert len(regions)==1
    assert regions[0]['appearance']!='neutral'
    assert regions[0]['axially_clipped']
    assert regions[0]['width_px']==pytest.approx(100,abs=2)


def test_rejected_fit_errors_expose_limits_without_exposing_a_measurement():
    from pipe_twin.metrology import CylinderFitError,fit_local_cylinder
    from test_metrology import cylinder
    points = cylinder()
    rng = np.random.default_rng(49)
    points += rng.normal(0,3,points.shape)
    with pytest.raises(CylinderFitError) as error:
        fit_local_cylinder(points)
    diagnostic = error.value.diagnostics
    assert str(error.value)=='CYLINDER_FIT_RESIDUAL_TOO_LARGE'
    assert diagnostic['fit_rms_mm']>diagnostic['maximum_fit_rms_mm']
    assert 'diameter_mm' not in diagnostic
    assert diagnostic['scope']=='REJECTED_FIT_NOT_MEASUREMENT'


def test_zone_attempts_have_global_coordinates_and_candidate_links():
    from pipe_twin.elevation_gui import recognition_diagnostic_lines
    cal,specs,groups = _groups()
    rng = np.random.default_rng(73)
    for role in ('left','right'):
        z = getattr(groups[0]['depth'],role+'_depth_mm')
        z[:] = rng.uniform(700,4000,z.shape)
    result = analyze_elevation_auto_zones(groups,calibration=cal,pipe_specs=specs,
        scope_settings={'scope':'zones','zones':[{'zone_id':'Z09','roi_rect_px':[120,100,390,240]}]},
        registration_settings={'local_observation_mode':'image_guided'})
    assert not result['local_surface']['observations']
    candidates = result['local_surface']['image_candidates']
    for candidate in candidates:
        for attempt in candidate['attempts']:
            for rejected in attempt['rejections']:
                assert rejected['image_candidate_id']==candidate['candidate_id']
                assert rejected['zone_id']=='Z09'
                if rejected.get('region_px'):
                    assert rejected['region_px'][1]>=100
                    if rejected['role']=='left':
                        assert rejected['region_px'][0]>=120
    assert all(r['zone_id']=='Z09' for r in result['local_surface']['rejected'])
    text='\n'.join(recognition_diagnostic_lines(result))
    assert '图像唯一匹配' in text and '主导深度支持' in text and 'Z09:IMG' in text
    assert candidates==result['capture_audit']['groups'][0]['image_candidates']
    json.dumps(result,allow_nan=False)
