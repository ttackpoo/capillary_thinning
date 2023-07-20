from pytest import raises
from mcplexpt.core import Experiment, ExperimentSuite
from mcplexpt.testing import get_samples_path

A_path = get_samples_path("core", "A.txt")
B_txt_path = get_samples_path("core", "B.txt")
B_csv_path = get_samples_path("core", "B.csv")
C_path = get_samples_path("core", "C.json")
D_path = get_samples_path("core", "D")
E_path = get_samples_path("core", "E.mp4")
F_path = get_samples_path("core", "F.mp4")

A_exp = Experiment(A_path)
B_txt_expt = Experiment(B_txt_path)
B_csv_expt = Experiment(B_csv_path)
C_exp = Experiment(C_path)
D_exp = Experiment(D_path)
E_exp = Experiment(E_path)
F_exp = Experiment(F_path)

samples_core_path = get_samples_path("core")
all_suite1 = ExperimentSuite(samples_core_path)
all_suite2 = ExperimentSuite(samples_core_path, param_file="D")
txt_suite1 = ExperimentSuite(samples_core_path, globs=["*.txt"])
txt_suite2 = ExperimentSuite(samples_core_path, globs=["*.txt"],
                             param_file="B.txt")
B_json_suite = ExperimentSuite(samples_core_path, globs=["B.*", "*.json"])
subdir_suite = ExperimentSuite(samples_core_path, globs=["subdir"])


def test_Experiment_construction():

    # File must exist
    with raises(TypeError):
        Experiment(get_samples_path("core", "E"))

    # Directory cannot be Experiment
    with raises(TypeError):
        Experiment(get_samples_path("core", "subdir"))

    assert A_exp.path == A_path
    assert A_exp.name == "A"
    assert A_exp.extension == "txt"

    assert B_txt_expt.name == "B"
    assert B_txt_expt.extension == "txt"

    assert B_csv_expt.name == "B"
    assert B_csv_expt.extension == "csv"

    assert D_exp.name == "D"
    assert D_exp.extension == ""

    assert E_exp.name == "E"
    assert E_exp.extension == "mp4"


def test_Experiment_eq():

    assert A_exp == Experiment(A_path)
    assert B_txt_expt == Experiment(B_txt_path)
    assert B_csv_expt == Experiment(B_csv_path)

    assert A_exp != B_txt_expt
    assert A_exp != B_csv_expt


def test_Experiment_repr():

    assert repr(A_exp) == "<Experiment 'A.txt'>"
    assert repr(B_txt_expt) == "<Experiment 'B.txt'>"
    assert repr(B_csv_expt) == "<Experiment 'B.csv'>"


def test_ExperimentSuite_construction():

    # Directory must exist
    with raises(TypeError):
        ExperimentSuite(get_samples_path("foo"), Experiment)

    # File cannot be ExperimentSuite
    with raises(TypeError):
        ExperimentSuite(A_path, Experiment)

    assert all_suite1.path == samples_core_path
    assert all_suite1.globs == ("*",)
    assert all_suite1.param_file == ""
    assert all_suite1.name == "core"

    assert all_suite2.param_file == "D"
    assert all_suite2.name == "core"

    assert txt_suite1.globs == ("*.txt",)

    # globs is sorted
    assert B_json_suite.globs == ("*.json", "B.*")


def test_ExperimentSuite_contents():

    assert all_suite1.contents == ['A.txt', 'B.csv', 'B.txt', 'C.json', 'D',
                                   'E.mp4', 'F.mp4', 'G.png']

    # Parameter file is not included
    assert all_suite2.contents == ['A.txt', 'B.csv', 'B.txt', 'C.json',
                                   'E.mp4', 'F.mp4', 'G.png']

    # Glob is matched
    assert txt_suite1.contents == ['A.txt', 'B.txt']

    # Parameter file is not matched by glob
    assert txt_suite2.contents == ['A.txt']

    # Multiple globs are matched
    assert B_json_suite.contents == ['B.csv', 'B.txt', 'C.json']

    # Subdirectory is never included, even if explicitly specified
    assert subdir_suite.contents == []


def test_ExperimentSuite_eq():

    assert all_suite1 == ExperimentSuite(samples_core_path)
    assert all_suite2 == ExperimentSuite(samples_core_path, param_file="D")
    assert txt_suite1 == ExperimentSuite(samples_core_path, globs=["*.txt"])
    assert txt_suite2 == ExperimentSuite(
        samples_core_path, globs=["*.txt"], param_file="B.txt"
    )
    assert B_json_suite == ExperimentSuite(samples_core_path, globs=["B.*", "*.json"])

    assert all_suite1 != all_suite2
    assert all_suite1 != B_json_suite
    assert txt_suite1 != txt_suite2


def test_ExperimentSuite_repr():

    assert repr(all_suite1) == "<Experiment Suite 'core'>"
    assert repr(all_suite2) == "<Experiment Suite 'core'>"
    assert repr(txt_suite1) == "<Experiment Suite 'core'>"
    assert repr(txt_suite2) == "<Experiment Suite 'core'>"
    assert repr(B_json_suite) == "<Experiment Suite 'core'>"
    assert repr(subdir_suite) == "<Experiment Suite 'core'>"


def test_ExperimentSuite_get():

    assert all_suite1.get("A") == A_exp
    assert all_suite1.get("C") == C_exp
    assert all_suite1.get("B.txt") == B_txt_expt
    assert all_suite1.get("B.csv") == B_csv_expt
