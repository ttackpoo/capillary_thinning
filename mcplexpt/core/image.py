"""실험 이미지 파일을 추상화하는 클래스를 제공하는 모듈입니다."""

import cv2
<<<<<<< HEAD
import numpy as np
import cupy as cp

=======
>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0

from .experiment import Experiment

__all__ = ["ExperimentImage", "ExperimentTIFF"]

<<<<<<< HEAD
=======

>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0
class ExperimentImage(Experiment):
    """
    실험 이미지 파일을 추상화하는 클래스입니다.

    본 클래스는 영상 분석에 유용한 메소드들을 제공합니다. 이미지 파일을 다루는 실험
    파일 클래스는 이 메소드를 상속받는 것이 권장됩니다.

    Examples
    ========

    >>> from mcplexpt.core import ExperimentImage
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("core", "G.png")
    >>> expt = ExperimentImage(path)
    >>> expt.shape
    (120, 160, 3)
    >>> expt.width
    160
    >>> expt.height
    120

    Attributes
    ==========

    frame : numpy.ndarray
        이미지 행렬입니다.

    shape : tuple of int
        이미지 행렬의 차원입니다.

    height, width : int
        이미지의 세로와 가로 픽셀 수 입니다.

    """
    def __init__(self, path):
        super().__init__(path)
<<<<<<< HEAD
        self.frame = cv2.cuda.imread(path)
        self.gpu_frame=cv2.GpuMat()
        self.gpu_frame.upload(self.frame)        
        self.height, self.width, _ = self.shape = self.gpu_frame.shape
=======
        self.frame = cv2.imread(path)
        self.height, self.width, _ = self.shape = self.frame.shape
>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0


class ExperimentTIFF(Experiment):
    """
    실험 TIFF 이미지 파일을 추상화하는 클래스입니다.

    TIFF 파일은 .tiff 확장자를 가지는, 하나 또는 여러 이미지의 묶음 파일입니다.

    .. warning::
        현재 이 클래스는 TIFF 파일의 모든 페이지를 한 번에 메모리에 로드합니다.
        각각의 페이지를 한 장씩 읽어오도록 하는 방법을 찾아 개선해야 합니다.

    Examples
    ========

    ``images`` 어트리뷰트는 파일의 모든 이미지를 리스트로 가져옵니다.
    :func:`get_nth_image() <ExperimentTIFF.get_nth_image>` 메소드를 사용해 n번째
    이미지를 가져올 수 있습니다.

    .. plot::
        :include-source:

        >>> import matplotlib.pyplot as plt
        >>> from mcplexpt.core import ExperimentTIFF
        >>> from mcplexpt.testing import get_samples_path
        >>> path = get_samples_path("caber", "dos", "sample_250fps.tiff")
        >>> expt = ExperimentTIFF(path)
        >>> len(expt.images)
        14
        >>> plt.imshow(expt.get_nth_image(-1), cmap='gray') # doctest: +SKIP

    Attributes
    ==========

<<<<<<< HEAD
    images : list of numpy.ndarrays
=======
    images : list of numpy.ndarray
>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0
        파일의 이미지들입니다.

    """
    def __init__(self, path):
<<<<<<< HEAD
        super().__init__(path)      
        ret, self.images = cv2.imreadmulti(path)
        self.gpu_images = []
        for img in self.images:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(img)       
            self.gpu_images.append(gpu_img)     
        
=======
        super().__init__(path)
        ret, self.images = cv2.imreadmulti(path)
>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0

    def get_nth_image(self, n):
        """
        TIFF 파일의 *n* 번째 이미지를 반환합니다.

        Parameters
        ==========

        n : int
            이미지 번호입니다. 파이썬의 인덱싱 규칙을 따릅니다.

        Returns
        =======

        image : numpy.ndarray
             *n* 번째 이미지의 행렬입니다.

        """
<<<<<<< HEAD
        return self.gpu_images[n]
=======
        return self.images[n]
>>>>>>> fb631c5eb8460ebedb5bb280d305a7259e0c1ee0
