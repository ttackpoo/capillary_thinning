"""실험 동영상 파일을 추상화하는 클래스를 제공하는 모듈입니다."""

import cv2
from functools import lru_cache
import itertools
import numpy as np

from .experiment import Experiment

__all__ = ["ExperimentVideo"]


class ExperimentVideo(Experiment):
    """
    실험 동영상 파일을 추상화하는 클래스입니다.

    본 클래스는 영상 분석에 유용한 메소드들을 제공합니다. 동영상 파일을 다루는 실험
    파일 클래스는 이 메소드를 상속받는 것이 권장됩니다.

    Examples
    ========

    >>> from mcplexpt.core import ExperimentVideo
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("core", "E.mp4")
    >>> expt = ExperimentVideo(path)
    >>> expt.fps
    1.0
    >>> expt.frames
    4
    >>> expt.height
    60
    >>> expt.width
    120

    Attributes
    ==========

    fps : float
        동영상의 초당 프레임 수 입니다.

    frames : int
        동영상의 총 프레임 길이입니다.

    height, width : int
        동영상의 세로와 가로 픽셀 수 입니다.

    """
    def __init__(self, path):
        super().__init__(path)
        cap = cv2.VideoCapture(self.path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    def get_nth_frame(self, n):
        """
        동영상의 *n* 번째 프레임을 반환합니다.

        Parameters
        ==========

        n : int
            프레임 번호입니다. 파이썬의 인덱싱 규칙을 따릅니다.

        Returns
        =======

        frame : numpy.ndarray
             *n* 번째 프레임의 이미지 행렬입니다.

        """
        cap = cv2.VideoCapture(self.path)
        lastframenum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if n < 0:
            framenum = lastframenum + n
        else:
            framenum = n
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenum)
        _, frame = cap.read()
        cap.release()
        return frame

    def frames_generator(self, frange=[0, -1], step=1):
        """
        동영상의 프레임들을 하나씩 yield하는 제네레이터를 반환합니다.

        Parameters
        ==========

        frange : tuple of two int, optional
            프레임 범위입니다. 파이썬의 인덱싱 규칙을 따릅니다. 명시되지 않을 시
            동영상의 전체 범위가 됩니다.

        step : positive int, default=1
            건너뛸 프레임 간격입니다. 1일 경우 건너뛰지 않습니다.

        Yields
        ======

        frame : numpy.ndarray

        """
        cap = cv2.VideoCapture(self.path)
        lastframenum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start, end = frange
        if start < 0:
            start = lastframenum + start + 1
        if end < 0:
            end = lastframenum + end + 1
        start, end = int(start), int(end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in range(start, end):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            if i % step != 0:
                continue
            yield frame

    @lru_cache(maxsize=8)
    def average_frames(self, frange=[0, -1]):
        """
        동영상의 모든 프레임들의 평균을 취한 하나의 프레임을 반환합니다.

        Parameters
        ==========

        frange : tuple of two int
            프레임 범위입니다. 파이썬의 인덱싱 규칙을 따릅니다. 명시되지 않을 시
            동영상의 전체 범위가 됩니다.

        Returns
        =======

        result : numpy.ndarray

        """
        cap = cv2.VideoCapture(self.path)
        lastframenum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        start, end = frange
        if start < 0:
            start = lastframenum + start + 1
        if end < 0:
            end = lastframenum + end + 1
        start, end = int(start), int(end)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for i in range(start, end):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        result = np.average(frames, axis=0).astype(np.uint8)
        return result

    def detect_blackbarwidth(self):
        """
        비디오 주위의 검은 영역의 두께를 구합니다.

        동영상 편집을 하다 보면 동영상의 크기를 맞추고자 프레임 주위에 검은 영역이
        덧대어지는 경우가 있습니다. 이 메소드는 해당 영역의 두께를 인식합니다.

        .. warning::
            동영상 자체의 가장자리에 검은 부분이 있을 경우, 해당 영역까지 인식될 수
            있습니다.

        Returns
        =======

        updown, leftright : int
            검은 영역의 두께에 해당하는 픽셀 수입니다.

        Examples
        ========

        >>> from mcplexpt.core import ExperimentVideo
        >>> from mcplexpt.testing import get_samples_path
        >>> path = get_samples_path("core", "E.mp4")
        >>> expt = ExperimentVideo(path)
        >>> expt.detect_blackbarwidth()
        (15, 30)

        """
        updowns, leftrights = [], []
        fgen = self.frames_generator()
        for frame in fgen:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, frame = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            updowns.append(self._detect_updown_blackbar(frame))
            leftrights.append(self._detect_leftright_blackbar(frame))
        updown = min(updowns)
        leftright = min(leftrights)
        return updown, leftright

    def _detect_updown_blackbar(self, frame):
        # helper function
        for i in range(frame.shape[0]):
            row = frame[i, :]
            if not np.all(row == 0):
                break
        for j in range(frame.shape[0]):
            j = frame.shape[0] - j - 1
            row = frame[j, :]
            if not np.all(row == 0):
                break
        return min([i, j])

    def _detect_leftright_blackbar(self, frame):
        # helper function
        for i in range(frame.shape[1]):
            col = frame[:, i]
            if not np.all(col == 0):
                break
        for j in range(frame.shape[1]):
            j = frame.shape[1] - j - 1
            col = frame[:, j]
            if not np.all(col == 0):
                break
        return min([i, j])

    def cropped_generator(self, x_roi=(0, -1), y_roi=(0, -1)):
        """
        동영상의 프레임들을 크롭해서 하나씩 yield하는 제네레이터를 반환합니다.

        Parameters
        ==========

        x_roi, y_roi : tuple of two int
            크롭 후 남길 가로 영역과 세로 영역입니다.

        Yields
        ======

        frame : numpy.ndarray

        """
        fgen = self.frames_generator()

        firstframe = next(fgen)
        dim = len(firstframe.shape)
        if dim == 3:
            color = True
        elif dim == 2:
            color = False
        else:
            raise ValueError("Dimension of the frame is %s" % dim)
        fgen = itertools.chain((firstframe,), fgen)

        y_slice = slice(y_roi[0], y_roi[1])
        x_slice = slice(x_roi[0], x_roi[1])
        for f in fgen:
            if color:
                f = f[y_slice, x_slice, :]
            else:
                f = f[y_slice, x_slice]
            yield f
