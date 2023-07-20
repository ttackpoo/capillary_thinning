import numpy as np
from mcplexpt.core import ExperimentImage, ExperimentTIFF
from mcplexpt.testing import get_samples_path

G_path = get_samples_path("core", "G.png")
img_G = ExperimentImage(G_path)

tiff_path = get_samples_path("caber", "dos", "sample_250fps.tiff")
tiff = ExperimentTIFF(tiff_path)

def test_ExperimentImage_attributes():
    assert (img_G.frame == np.zeros(shape=(120, 160, 3))).all()
    assert img_G.shape == (120, 160, 3)
    assert img_G.width == 160
    assert img_G.height == 120


def test_ExperimentTIFF_attributes():
    assert len(tiff.images) == 14
