"""Image ambiguity diagnostics must remain independent of model/measurements."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.stereo_correspondence import patch_correspondence


def scene(kind="random"):
    rng = np.random.default_rng(61)
    h, w, d = 180, 400, 20
    if kind == "random":
        texture = cv2.GaussianBlur(rng.integers(30,210,(h,w+d),np.uint8),(3,3),.6)
    elif kind == "repeat":
        texture = np.tile((100+50*np.sin(np.arange(w+d)*2*np.pi/8)).astype(np.uint8),(h,1))
    else:
        texture = np.tile(np.arange(h,dtype=np.uint8)[:,None],(1,w+d))
    left = cv2.cvtColor(texture[:,:w],cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(np.clip(texture[:,d:]*1.05+5,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
    camera = SimpleNamespace(rectified_intrinsic=np.array([[500.,0,200],[0,500,90],[0,0,1]]))
    depth = np.full((h,w),1500.,np.float32)
    valid = np.ones((h,w),bool)
    region = {"region_px":[90,25,200,120],"mask":np.ones((120,200),np.uint8)}
    return left,right,depth,valid,camera,region


@pytest.mark.parametrize("role",["left","right"])
def test_known_disparity_has_unique_matches_without_changing_input(role):
    left,right,z,valid,camera,region = scene()
    a,b = (left,right) if role == "left" else (right,left)
    before = [x.copy() for x in (a,b,z,valid)]
    result = patch_correspondence(a,b,z,valid,camera,camera,60,role,region)
    assert result["sampled_unique_match_fraction"] > .95
    assert result["sample_count"] <= 512
    assert not result["measurement_authority"]
    assert not result["warning_codes"]
    for x,y in zip(before,(a,b,z,valid)):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize("kind,warning",[("repeat","AMBIGUOUS_PATCH_CORRESPONDENCE"),
                                         ("flat","LOW_HORIZONTAL_TEXTURE")])
def test_high_correlation_is_not_unique_depth(kind,warning):
    left,right,z,valid,camera,region = scene(kind)
    result = patch_correspondence(left,right,z,valid,camera,camera,60,"left",region)
    assert result["correlation_median"] > .99
    assert result["sampled_unique_match_fraction"] == 0
    assert warning in result["warning_codes"]


def test_wrong_disparity_does_not_become_unique_by_photometric_agreement():
    left,right,z,valid,camera,region = scene()
    z[:] = 500*60/36
    result = patch_correspondence(left,right,z,valid,camera,camera,60,"left",region)
    assert result["sampled_unique_match_fraction"] == 0
    assert "LOW_PATCH_CORRELATION" in result["warning_codes"]


@pytest.mark.parametrize("role",["left","right"])
def test_asymmetric_crop_preserves_same_physical_correspondences(role):
    left,right,z,valid,camera,region = scene()
    a,b = (left,right) if role == "left" else (right,left)
    full = patch_correspondence(a,b,z,valid,camera,camera,60,role,region)
    k1,k2 = camera.rectified_intrinsic.copy(),camera.rectified_intrinsic.copy()
    k1[0,2] -= 37; k2[0,2] -= 11
    local = {**region,"region_px":[53,25,200,120]}
    cropped = patch_correspondence(a[:,37:],b[:,11:],z[:,37:],valid[:,37:],
        SimpleNamespace(rectified_intrinsic=k1),SimpleNamespace(rectified_intrinsic=k2),60,role,local)
    for key in ("correlation_median","uniqueness_margin_median","sampled_unique_match_fraction"):
        assert cropped[key] == pytest.approx(full[key],abs=1e-5)


def test_missing_depth_cannot_report_zero_ambiguity_as_good():
    left,right,z,valid,camera,region = scene()
    valid[:] = False
    result = patch_correspondence(left,right,z,valid,camera,camera,60,"left",region)
    assert result["status"] == "INSUFFICIENT_SAMPLES"
    assert "sampled_unique_match_fraction" not in result
