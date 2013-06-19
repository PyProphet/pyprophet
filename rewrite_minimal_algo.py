import os
# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
os.putenv("OPENBLAS_NUM_THREADS", "1")


try:
    profile
except:
    profile = lambda x: x


from pyprophet.pyprophet import PyProphet
from pyprophet.config    import standard_config


@profile
def main():

    path = "orig_r_code/test_reduced.txt"
    config = standard_config()
    summary_table, final_table = PyProphet().process_csv(path, "\t", config)
    print summary_table
    return


if __name__ == "__main__":
    main()
