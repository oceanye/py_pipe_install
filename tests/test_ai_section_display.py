from pipe_twin.ai_section_display import ai_section_summary_lines, section_candidate_rows


def test_empty_namespaces_keep_not_run_rejected_and_insufficient_distinct():
    for status in ('NOT_RUN', 'REJECTED', 'INSUFFICIENT'):
        report = {'ai_section_assist': {'scope': 'CROSS_SECTION_CANDIDATE_ONLY',
                                       'status': status, 'candidates': []}}
        assert section_candidate_rows(report)[0]['status'] == status


def test_pixel_control_section_is_not_presented_as_millimetres():
    report = {'ai_section_assist': {'scope': 'CROSS_SECTION_CANDIDATE_ONLY', 'status': 'UNRESOLVED',
              'control_section': {'status': 'DEFINED', 'coordinate_space': 'rectified_left_pixel',
                                  'station_px': [300, 240], 'line_px': [[300, 0], [300, 480]]}}}
    text = '\n'.join(ai_section_summary_lines(report))
    assert '图像控制断面' in text
    assert '像素' in text
    assert '300.0 mm' not in text


def test_live_result_details_distinguish_capture_time_and_inference_time():
    report = {'ai_section_assist': {'scope': 'CROSS_SECTION_CANDIDATE_ONLY', 'status': 'UNRESOLVED',
              'model': 'kimi-k3.1-pro', 'generated_at': '2026-11-01T02:00:00+00:00',
              'observation_scope': 'LATEST_REFERENCE_PAIR_ONLY_NOT_TEMPORAL_FUSION',
              'asset_refs': [{'kind': 'FIELD_IMAGE', 'captured_at': '2026-10-30T18:32:50+08:00'}]}}
    text = '\n'.join(ai_section_summary_lines(report))
    assert 'kimi-k3.1-pro' in text
    assert '结果生成时间：2026-11-01' in text
    assert '实拍时间：2026-10-30' in text
    assert '仅当前参考双目对' in text
    assert '不是重新采集' in text


def test_temporal_ai_result_and_metric_control_section_are_explicitly_labeled():
    report = {'ai_section_assist': {
        'scope': 'SECTION_ONLY', 'status': 'UNRESOLVED',
        'observation_scope': 'TEMPORAL_BURST_REFERENCE_SET',
        'temporal_frame_count': 10,
        'control_section': {
            'status': 'DEFINED', 'coordinate_space': 'model_axis_mm',
            'station_mm': -141.676, 'interval_mm': [-296.06, 12.71],
        },
        'pooled_control_section': {
            'status': 'DEFINED', 'coordinate_space': 'model_axis_mm',
            'station_mm': -141.676, 'interval_mm': [-296.06, 12.71],
        },
    }}
    text = '\n'.join(ai_section_summary_lines(report))
    assert '整组连续双目图像（10 组代表性双目帧）' in text
    assert 'AI 输入控制断面：公共支撑区间中点 -141.7 mm' in text
    assert 'AI 已绑定的主流程控制断面：公共支撑区间中点 -141.7 mm' in text


def test_summary_exposes_full_replay_evidence_without_turning_it_into_ai_state():
    report = {
        'ai_section_assist': {
            'scope': 'SECTION_ONLY', 'status': 'UNRESOLVED',
            'control_section': {
                'status': 'UNRESOLVED',
                'reason_codes': ['AI_AGGREGATION_BAND_MISSING'],
            },
        },
        'temporal_fusion': {
            'status': 'VALID', 'requested_frames': 10, 'accepted_frames': 10,
        },
        'section_evaluation': {
            'status': 'ANALYZED', 'conditional_section_count': 2,
            'candidates': [{}, {}, {}, {}],
            'control_section': {
                'status': 'DEFINED', 'station_mm': -141.676,
                'interval_mm': [-296.06, 12.71],
            },
        },
        'local_surface': {
            'observations': [], 'image_candidates': [{}, {}],
            'audit': {'metric_cylinder_count': 0},
        },
    }
    text = '\n'.join(ai_section_summary_lines(report))
    assert '时序融合：已通过（10/10 帧）' in text
    assert '断面汇聚：已分析，候选 4 个，条件候选 2 个' in text
    assert '可供 AI 引用的主流程控制断面' in text
    assert '局部双目证据：可用观测 0，图像候选 2，可匹配圆柱 0' in text
    assert 'AI 输入控制断面：未稳定确定' in text
