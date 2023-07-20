from mcplexpt.caber.dos import DoSCaBERExperiment
from mcplexpt.caber.dos.dos import DoSCaBERError
from mcplexpt.testing import get_samples_path
path = get_samples_path("caber", "dos", "sample_250fps.tiff")
expt = DoSCaBERExperiment(path)
import pytest

def test_DoSCaBERExperiment_capbridge_broken():
    ret = []
    for img in expt.images:
        ret.append(expt.capbridge_broken(img))
    corrrect_ret = ([False] * 11) + ([True]*3)
    assert ret == corrrect_ret

def test_DoSCaBERExperiment_width_average_fixedroi():
    ret = []
    for img in expt.images[0:2]:
        with pytest.raises(DoSCaBERError):
            expt.width_average_fixedroi(img)
    for img in expt.images[2:14]:
        ret.append(expt.width_average_fixedroi(img))
    corrrect_ret = ([182])+([23])+([14])+([9])+([6])+([5])+([3])+([2])+([2])+([1])+([1])+([3])
    assert ret == corrrect_ret

def test_DoSCaBERExperiment_width_average_minimumroi():
    ret = []
    for img in expt.images[0:2]:
        with pytest.raises(DoSCaBERError):
            expt.width_average_minimumroi(img)
    for img in expt.images[2:14]:
        ret.append(expt.width_average_minimumroi(img))
    corrrect_ret = ([128])+([22])+([14])+([9])+([6])+([4])+([2])+([2])+([1])+([0])+([1])+([2])
    assert ret == corrrect_ret