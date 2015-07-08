import pdb
from pyprophet.main import _main
import os

def test(tmpdir):
    here = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(here, "test_data.txt")
    _main(["--target.dir=%s" % tmpdir.strpath, "--target.overwrite", in_file])
    for f in ["test_data_summary_stat.csv", "test_data_full_stat.csv", "test_data_report.pdf",
            "test_data_scorer.bin",
            "test_data_cutoffs.txt",
            "test_data_svalues.txt",
            "test_data_qvalues.txt",
            "test_data_dscores_top_target_peaks.txt",
            "test_data_dscores_top_decoy_peaks.txt",
            ]:
        full_path = os.path.join(tmpdir.strpath, f)
        assert os.path.exists(full_path)


def test_with_probability(tmpdir):
    here = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(here, "test_data.txt")
    _main(["--target.dir=%s" % tmpdir.strpath, "--target.overwrite --compute.probabilities", in_file])
    for f in ["test_data_summary_stat.csv", "test_data_full_stat.csv", "test_data_report.pdf",
            "test_data_scorer.bin",
            "test_data_cutoffs.txt",
            "test_data_svalues.txt",
            "test_data_qvalues.txt",
            "test_data_dscores_top_target_peaks.txt",
            "test_data_dscores_top_decoy_peaks.txt",
            ]:
        full_path = os.path.join(tmpdir.strpath, f)
        assert os.path.exists(full_path)

