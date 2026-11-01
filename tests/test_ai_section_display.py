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
