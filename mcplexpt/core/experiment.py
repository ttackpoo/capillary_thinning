"""실험 데이터와 디렉토리를 추상화하는 가장 기초적인 클래스를 제공하는 모듈입니다."""

__all__ = ["Experiment", "ExperimentSuite"]

import glob, os, re


class Experiment(object):
    """
    실험 데이터 파일을 추상화하는 클래스입니다.

    본 클래스는 파일 경로와 확장자를 인식하는 기능을 지원합니다. 데이터 분석
    알고리즘은 서브클래스의 메소드로 구현됩니다.

    .. warning::
        본 클래스는 직접 호출되어서는 안 됩니다. 실험 변수의 올바른 적용을 위해,
        인스턴스의 생성은 실험 디렉토리 클래스의 ``get()`` 메소드를 통해
        이루어져야 합니다.

    Parameters
    ==========

    path : str
        실험 데이터 파일로의 경로입니다.

    Examples
    ========

    먼저 실험 디렉토리 경로를 이용해 :class:`ExperimentSuite()` 인스턴스를
    생성합니다. 그리고 :func:`get()<ExperimentSuite.get>` 메소드를 이용해
    ``Experiment()`` 인스턴스를 생성합니다.

    >>> from mcplexpt.core import ExperimentSuite
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("core")
    >>> suite = ExperimentSuite(path)
    >>> expt_A = suite.get('A')
    >>> expt_B = suite.get('B.csv')

    파일의 이름과 확장자가 자동으로 구분됩니다.

    >>> expt_A.name
    'A'
    >>> expt_A.extension
    'txt'
    >>> expt_B.name
    'B'
    >>> expt_B.extension
    'csv'

    Attributes
    ==========

    path : str
        실험 데이터 파일로의 경로입니다.

    name : str
        실험 데이터 파일명입니다. 확장자는 제외됩니다.

    extension : str
        실험 데이터 파일의 확장자입니다.

    """

    def __init__(self, path):
        if not os.path.isfile(path):
            raise TypeError("%s is not a file." % path)

        self.path = path
        self.name = os.path.split(path)[-1].split(os.extsep)[0]
        self.extension = "".join(os.path.split(path)[-1].split(os.extsep)[1:])

    def __eq__(self, other):
        return type(self) == type(other) and self.path == other.path

    def __repr__(self):
        clsname = " ".join(re.findall("[A-Z][^A-Z]*", type(self).__name__))
        name = os.path.extsep.join([self.name, self.extension])
        return "<%s '%s'>" % (clsname, name)


class ExperimentSuite(object):
    """
    실험 데이터 디렉토리를 추상화하는 클래스입니다.

    본 클래스는 디렉토리 안의 실험 데이터 파일과 변수 파일을 인식하고, 알맞은 실험
    파일 클래스를 생성할 수 있습니다. 실험 파일 클래스의 등록 및 변수 파일 규칙은
    서브클래스에서 구현되어야 합니다.

    Parameters
    ==========

    path : str
        실험 데이터 디렉토리로의 경로입니다.

    globs : list of str, default=['*']
        실험 파일들의 glob 패턴입니다. 주어지지 않을 시 모든 파일이 실험 파일로
        인식됩니다.

    param_file : str, optional
        변수 파일명입니다. 확장자가 명시되어야 합니다.

    Notes
    =====

    본 클래스는 디렉토리 안의 파일만을 인식합니다. 하위 디렉토리는 무시됩니다.

    Examples
    ========

    본 패키지의 `samples/core` 디렉토리를 예시로 듭니다. 해당 경로에는 다양한
    확장자의 여러 다른 파일들이 존재합니다.

    >>> from mcplexpt.core import ExperimentSuite
    >>> from mcplexpt.testing import get_samples_path
    >>> path = get_samples_path("core")
    >>> suite1 = ExperimentSuite(path)
    >>> suite1
    <Experiment Suite 'core'>
    >>> suite1.contents
    ['A.txt', 'B.csv', 'B.txt', 'C.json', 'D', 'E.mp4', 'F.mp4', 'G.png']

    ``get()`` 메소드를 이용해 실험 파일 클래스를 생성할 수 있습니다.

    >>> suite1.get('A')
    <Experiment 'A.txt'>
    >>> suite1.get('B.csv')
    <Experiment 'B.csv'>

    특정 파일들만이 데이터 파일로 인식되도록 규칙을 명시할 수 있습니다. 또한 변수
    파일을 지정해 줄 수 있으며, 이 때 변수 파일은 데이터 파일로 인식되지 않습니다.

    >>> suite2 = ExperimentSuite(path, globs=['*.txt'], param_file='A.txt')
    >>> suite2.contents
    ['B.txt']
    >>> suite2.param_file
    'A.txt'
    >>> suite2.get('B')
    <Experiment 'B.txt'>

    Attributes
    ==========

    contents : list of str
        디렉토리에 포함된 실험 데이터 파일들의 파일명입니다.

    param_file : str
        변수 파일의 파일명입니다.

    """

    def __init__(self, path, globs=["*"], param_file=""):
        if not os.path.isdir(path):
            raise TypeError("%s is not a directory." % path)

        self.path = path
        self.globs = tuple(sorted(globs))
        self.param_file = param_file

        self.name = os.path.split(path)[-1]

        self.contents = []
        for pattern in globs:
            files = glob.glob(os.path.join(path, pattern))
            for f in files:
                # subdirectory is not added
                if not os.path.isfile(f):
                    continue
                fname = os.path.split(f)[-1]
                # do not include parameter file
                if fname == param_file:
                    continue
                self.contents.append(fname)
        self.contents.sort()

    def search_path(self, expt_name):
        """
        *expt_name* 에 해당하는 파일의 경로를 반환합니다.

        Parameters
        ==========

        expt_name : str
            실험 파일명입니다.

        Returns
        =======

        path : str
            실험 파일의 전체 경로입니다.

        Raises
        ======

        ValueError
            해당하는 파일이 없거나, 둘 이상의 파일이 해당됩니다.

        """
        matches = []
        for fname in self.contents:
            fpath = os.path.join(self.path, fname)
            if expt_name == fname:
                # matches full name
                return fpath
            if expt_name == os.path.splitext(fname)[0]:
                # matches name without extension
                matches.append(fpath)
        if len(matches) == 0:
            raise ValueError(
                "No file in %s matches to '%s'." % (self, expt_name)
            )
        elif len(matches) > 2:
            raise ValueError(
                "Multiple files in %s matche to '%s'." % (self, expt_name)
            )
        return matches[0]

    def get(self, expt_name):
        """
        실험 파일 인스턴스를 반환합니다.

        실험 변수가 정의된 서브클래스의 경우, 반드시 이 메소드를 재정의해서 실험
        변수가 올바르게 적용되도록 하여야 합니다.

        Parameters
        ==========

        expt_name : str
            실험 파일명입니다.

        Returns
        =======

        Experiment
            변수가 올바르게 적용된 실험 파일 인스턴스입니다.

        """
        path = self.search_path(expt_name)
        return Experiment(path)

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        if not self.globs == other.globs:
            return False
        if not self.param_file == other.param_file:
            return False
        return True

    def __repr__(self):
        clsname = " ".join(re.findall("[A-Z][^A-Z]*", type(self).__name__))
        return "<%s '%s'>" % (clsname, self.name)
