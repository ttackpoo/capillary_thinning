import math
import numpy as np
from pytest import raises
from mcplexpt.core import ExperimentVideo
from mcplexpt.testing import get_samples_path

E_path = get_samples_path("core", "E.mp4")
F_path = get_samples_path("core", "F.mp4")

vid_E = ExperimentVideo(E_path)
vid_F = ExperimentVideo(F_path)


def test_ExperimentVideo_get_nth_frame():

    fgen = vid_E.frames_generator()
    for i, f in enumerate(fgen):
        assert np.all(f == vid_E.get_nth_frame(i))

    fgen = vid_E.frames_generator()
    for i, f in enumerate(reversed(list(fgen))):
        assert np.all(f == vid_E.get_nth_frame(-i - 1))


def test_ExperimentVideo_frames_generator():

    fgen0 = vid_E.frames_generator()
    assert np.all(next(fgen0) == vid_E.get_nth_frame(0))
    assert np.all(next(fgen0) == vid_E.get_nth_frame(1))
    assert np.all(next(fgen0) == vid_E.get_nth_frame(2))
    assert np.all(next(fgen0) == vid_E.get_nth_frame(3))
    with raises(StopIteration):
        next(fgen0)

    fgen1 = vid_E.frames_generator((0, 0))
    with raises(StopIteration):
        next(fgen1)

    fgen2 = vid_E.frames_generator((0, 3))
    assert np.all(next(fgen2) == vid_E.get_nth_frame(0))
    assert np.all(next(fgen2) == vid_E.get_nth_frame(1))
    assert np.all(next(fgen2) == vid_E.get_nth_frame(2))
    with raises(StopIteration):
        next(fgen2)

    fgen3 = vid_E.frames_generator((0, -1))
    assert np.all(next(fgen3) == vid_E.get_nth_frame(0))
    assert np.all(next(fgen3) == vid_E.get_nth_frame(1))
    assert np.all(next(fgen3) == vid_E.get_nth_frame(2))
    assert np.all(next(fgen3) == vid_E.get_nth_frame(3))
    with raises(StopIteration):
        next(fgen3)

    fgen4 = vid_E.frames_generator((0, 5))
    assert np.all(next(fgen4) == vid_E.get_nth_frame(0))
    assert np.all(next(fgen4) == vid_E.get_nth_frame(1))
    assert np.all(next(fgen4) == vid_E.get_nth_frame(2))
    assert np.all(next(fgen4) == vid_E.get_nth_frame(3))
    with raises(StopIteration):
        next(fgen4)

    fgen5 = vid_E.frames_generator((1, -2))
    assert np.all(next(fgen5) == vid_E.get_nth_frame(1))
    assert np.all(next(fgen5) == vid_E.get_nth_frame(2))
    with raises(StopIteration):
        next(fgen5)

    fgen6 = vid_E.frames_generator(step=2)
    assert np.all(next(fgen6) == vid_E.get_nth_frame(0))
    assert np.all(next(fgen6) == vid_E.get_nth_frame(2))
    with raises(StopIteration):
        next(fgen6)


def test_ExperimentVideo_detect_blackbarwidth():
    real_blackbar_val = (15, 30)

    vid_E_blackbar = vid_E.detect_blackbarwidth()
    for a, b in zip(vid_E_blackbar, real_blackbar_val):
        assert math.isclose(a, b, abs_tol=5)

    vid_F_blackbar = vid_F.detect_blackbarwidth()
    for a, b in zip(vid_F_blackbar, real_blackbar_val):
        assert math.isclose(a, b, abs_tol=5)
