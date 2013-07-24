NUM_XVAL=20
NUM_FRACTION=0.5
NUM_SEMISV_ITER=5


pyprophet --xeval.num_processes=10 \
          --xeval.num_iter=$NUM_XVAL\
          --xeval.fraction=$NUM_FRACTION\
          --target.overwrite=1\
          --semi_supervised_learner.num_iter=$NUM_SEMISV_ITER\
          --is_test=0\
          ../orig_r_code/testfile.csv


cp ../orig_r_code/testfile_with_dscore.csv ../orig_r_code/testfile_with_dscore2.csv

pyprophet --xeval.num_processes=1 \
          --xeval.num_iter=$NUM_XVAL\
          --xeval.fraction=$NUM_FRACTION\
          --target.overwrite=1\
          --semi_supervised_learner.num_iter=$NUM_SEMISV_ITER\
          --is_test=0\
          ../orig_r_code/testfile.csv
