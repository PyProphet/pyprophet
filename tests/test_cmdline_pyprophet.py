import pdb
from pyprophet.main import _main
import os

def test(tmpdir):
    here = os.path.dirname(os.path.abspath(__file__))
    in_file = os.path.join(here, "test_data.txt")
    _main(["--target.dir=%s" % tmpdir.strpath, "--target.overwrite", in_file])
    for f in ["test_data_summary_stat.csv", "test_data_full_stat.csv", "test_data_report.pdf",
            "test_data_scorer.bin"]:
        full_path = os.path.join(tmpdir.strpath, f)
        assert os.path.exists(full_path)
