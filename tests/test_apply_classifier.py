import os
import subprocess
import os
import shutil

__here__ = os.path.dirname(os.path.abspath(__file__))


def test_apply_classifier(tmpdir):

    os.chdir(tmpdir.strpath)
    data_path = os.path.join(__here__, "test_data.txt")
    shutil.copy(data_path, tmpdir.strpath)
    stdout = subprocess.check_output("pyprophet test_data.txt", shell=True,
            stderr=subprocess.STDOUT)

    # collect m score stats
    m_score_stat = [l for l in stdout.split("\n") if "mean m_score" in l]
    assert len(m_score_stat) == 1

    # split away log time etc:
    __, __, interesting_m_score_output = m_score_stat[0].partition("mean m_score")

    # collect s value stats
    s_value_stat = [l for l in stdout.split("\n") if "mean s_value" in l]
    assert len(s_value_stat) == 1

    # split away log time etc:
    __, __, interesting_s_value_output = s_value_stat[0].partition("mean s_value")

    stdout = subprocess.check_output(
                        "pyprophet test_data.txt --apply_scorer=test_data_scorer.bin --target.overwrite",
                        shell=True,
                        stderr=subprocess.STDOUT)

    # collect m score stats
    m_score_stat = [l for l in stdout.split("\n") if "mean m_score" in l]
    assert len(m_score_stat) == 1

    # split away log time etc:
    __, __, interesting_m_score_output2 = m_score_stat[0].partition("mean m_score")

    # collect s value stats
    s_value_stat = [l for l in stdout.split("\n") if "mean s_value" in l]
    assert len(s_value_stat) == 1

    # split away log time etc:
    __, __, interesting_s_value_output2 = s_value_stat[0].partition("mean s_value")

    assert interesting_m_score_output == interesting_m_score_output2
    assert interesting_s_value_output == interesting_s_value_output2


if __name__ == "__main__":
    pass

