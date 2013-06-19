#R --slave --args bin_dir=. data_file=testdata_times_4.tsv  num_xval=5< mProphetFixed.R
R --slave --args bin_dir=. data_file=test_reduced.txt  num_xval=1< mProphetFixed.R
