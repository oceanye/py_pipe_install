from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pipe_twin.calibration_wizard import (
    BoardObservation, BoardStabilityGate, ChessboardCalibrationError,
    ChessboardWizardDialog,
)


def observations(dx=0., right_dx=None, detector="sb"):
    xy=np.mgrid[:8,:6].reshape(2,-1).T.astype(float)*25 + [200,150]
    return {role: BoardObservation(xy+[offset,0],1000.,(1,1),detector)
            for role,offset in (("left",dx),("right",dx-50 if right_dx is None else right_dx))}


def test_stationary_board_needs_three_observations_and_dwell():
    gate=BoardStabilityGate()
    assert not gate.observe(observations(),10.)
    assert not gate.observe(observations(.08),10.4)
    assert gate.observe(observations(-.05),10.9)
    gate.reset()
    assert not gate.observe(observations(),11.)


@pytest.mark.parametrize("mode", ["both", "right_only", "slow_drift", "one_corner"])
def test_motion_in_either_eye_prevents_acceptance(mode):
    gate=BoardStabilityGate()
    for i in range(20):
        if mode=="right_only":
            obs=observations(0.,right_dx=-50+i*3.)
        elif mode=="one_corner":
            obs=observations()
            obs['right'].corners_px[0,0]+=i*2.
        else:
            obs=observations(i*(.3 if mode=="slow_drift" else 3.))
        assert not gate.observe(obs,10.+i*.2)


@pytest.mark.parametrize("interruption", ["missing_eye", "time_gap", "detector_change", "nonfinite"])
def test_interruption_restarts_the_full_dwell(interruption):
    gate=BoardStabilityGate()
    gate.observe(observations(),10.)
    assert not gate.observe(observations(),10.4)
    obs=observations()
    if interruption=="missing_eye":
        del obs['right']
    elif interruption=="detector_change":
        obs=observations(detector='classic')
    elif interruption=="nonfinite":
        obs['left'].corners_px[0,0]=np.nan
    now=16. if interruption=="time_gap" else 10.9
    assert not gate.observe(obs,now)
    assert not gate.observe(observations(),now+.2)
    assert not gate.observe(observations(),now+.4)
    assert gate.observe(observations(),now+1.1)


def test_stationary_board_passes_at_measured_high_resolution_gui_cadence():
    gate=BoardStabilityGate()
    assert not gate.observe(observations(),10.)
    assert not gate.observe(observations(.06),12.6)
    assert gate.observe(observations(-.06),15.2)


def test_auto_capture_does_not_admit_a_moving_board_with_sharp_background():
    dialog=object.__new__(ChessboardWizardDialog)
    dialog._capture_stability=BoardStabilityGate()
    dialog.continuous=SimpleNamespace(get=lambda:True)
    dialog.sharpness_gate=SimpleNamespace(get=lambda:'60')
    dialog.message=Mock()
    dialog.pairs=[]
    dialog._last_auto_time=0.
    dialog._last_left_observation=None
    dialog._pattern=lambda:(8,6)
    dialog._accept_pair=Mock()
    pair=SimpleNamespace(left=np.zeros((480,640,3),np.uint8),right=np.zeros((480,640,3),np.uint8))
    for t,dx in ((10.,0.),(10.4,4.),(10.9,8.),(11.3,12.),(11.7,12.),(12.2,12.)):
        with patch('pipe_twin.calibration_wizard.time.monotonic',return_value=t):
            dialog._maybe_auto_capture(pair,observations(dx))
        assert dialog._accept_pair.call_count == (1 if t==12.2 else 0)


def test_detector_change_retains_existing_pairs_ids_and_archive():
    dialog=object.__new__(ChessboardWizardDialog)
    old=observations()
    original=(old['left'],old['right'])
    dialog.pairs=[original]
    dialog._pair_capture_ids=[42]
    dialog.capture_archive=Mock()
    dialog._last_left_observation=old['left']
    incoming=observations(50.,detector='classic')
    with pytest.raises(ChessboardCalibrationError,match='已保留原有 1 组'):
        dialog._accept_pair(incoming['left'],incoming['right'],automatic=False)
    assert len(dialog.pairs)==1 and dialog.pairs[0] is original
    assert dialog._pair_capture_ids==[42]
    assert dialog._last_left_observation is old['left']
    assert dialog.capture_archive.mock_calls==[]


def test_manual_button_cannot_bypass_stability_and_works_after_preview_settles():
    dialog=object.__new__(ChessboardWizardDialog)
    pair=SimpleNamespace(left=np.zeros((480,640,3),np.uint8),
        right=np.ones((480,640,3),np.uint8),sync_delta_ms=0.)
    dialog.session=SimpleNamespace(read_pair=lambda:pair)
    dialog._opening=False
    dialog.max_sync=object()
    dialog._float_value=lambda *args,**kwargs:1.
    dialog._pattern=lambda:(8,6)
    dialog._sb_disabled_roles=set()
    dialog._capture_stability=BoardStabilityGate()
    dialog.sharpness_gate=SimpleNamespace(get=lambda:'60')
    dialog._accept_pair=Mock()
    dialog.pairs=[]
    dialog.app=SimpleNamespace(messagebox=Mock())
    dialog.window=Mock()
    obs=observations()
    with patch('pipe_twin.calibration_wizard.detect_board_corners',side_effect=list(obs.values())), \
         patch('pipe_twin.calibration_wizard.time.monotonic',return_value=10.):
        dialog.capture_pair()
    dialog._accept_pair.assert_not_called()
    assert '尚未停稳' in dialog.app.messagebox.showerror.call_args.args[1]
    dialog._capture_stability.observe(obs,11.)
    dialog._capture_stability.observe(obs,11.4)
    with patch('pipe_twin.calibration_wizard.detect_board_corners',side_effect=list(obs.values())), \
         patch('pipe_twin.calibration_wizard.time.monotonic',return_value=12.):
        dialog.capture_pair()
    dialog._accept_pair.assert_called_once()
