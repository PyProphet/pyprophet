import pdb
try:
    profile
except:
    profile = lambda x: x

import pyprophet.data_structures as ds
import pyprophet.stats
import numpy as np
import pandas as pd

import sklearn.lda

PLOT = True
PLOT = 0

@profile
def main():
    import pylab
    load_from = "old_code/test_reduced.txt"
    exp = ds.Experiment.from_csv(load_from, sep="\t")
    print exp.df
    if PLOT:
        __, top_target_scores = exp.get_top_main_scores_target()
        __, top_decoy_scores = exp.get_top_main_scores_decoy()
        pylab.figure()
        pylab.hist(top_decoy_scores, bins=20, label="decoy")
        pylab.hist(top_target_scores, bins=20, label="sure_target")
        pylab.legend()

    #random.seed(0)

    ws = []

    collected = []

    for k in range(1):

        # split data in test and eval sets:
        train_df, eval_df = exp.split(0.5)

        # before semi supervised iteration:
        # find indices of top ranked peaks in respect of main score
        # find indices of tap ranked decoys in respect of main score
        top_target_indices, top_target_scores = train_df.get_top_main_scores_target()
        top_decoy_indices, top_decoy_scores = train_df.get_top_main_scores_decoy()
        __, decoy_scores = train_df.get_top_main_scores_decoy()

        # find cutoff fdr from scores:
        cutoff, errt =  pyprophet.stats.find_cutoff(top_target_scores, decoy_scores, 0.4, 0.15)
        print cutoff

        # only take best target peaks:
        best_indices =[ i for (i, s) in zip(top_target_indices, top_target_scores) if s >= cutoff]

        # learn lda score from decoy peaks and best target peaks
        Xdecoy = train_df.get_feature_matrix_without_main_score(top_decoy_indices)
        Xtarget_sure = train_df.get_feature_matrix_without_main_score(best_indices)
        X = np.vstack((Xdecoy, Xtarget_sure))

        y = np.zeros((X.shape[0],))
        n_decoy = Xdecoy.shape[0]
        n_target = Xtarget_sure.shape[0]
        y[n_decoy:] = 1

        learner = sklearn.lda.LDA()
        model = learner.fit(X, y)
        w = model.scalings_.flatten()

        print "w=", w


        clf_scores = np.dot(train_df.get_full_feature_matrix_without_main_score(), w)
        print "mean=", np.mean(clf_scores), np.std(clf_scores, ddof=1)
        clf_scores -= np.mean(clf_scores)
        train_df.add_classifier_score(clf_scores)

        # semi supervised iteration:
        for inner in range(2):

            # rerank in respect of this lda score
            train_df.df.sort(("is_decoy", "tg_id", "classifier_score"),
                    ascending=(False, True, False), inplace=True)


            # for next iteration: find best target peaks in respect of
            # lda top ranked target peaks
            top_target_indices, top_target_scores = train_df.get_top_classifier_scores_target()
            top_decoy_indices, top_decoy_scores = train_df.get_top_classifier_scores_decoy()
            decoy_scores = train_df.get_main_scores_decoy()

            cutoff, errt =  pyprophet.stats.find_cutoff(top_target_scores,
                    top_decoy_scores, 0.4, 0.02)

            best_indices =[ i for (i, s) in zip(top_target_indices,
                top_target_scores) if s > cutoff]

            # learn lda from those peaks:
            Xdecoy = train_df.get_feature_matrix_with_main_score(top_decoy_indices)
            Xtarget = train_df.get_feature_matrix_with_main_score(best_indices)

            n_decoy = Xdecoy.shape[0]
            n_target = Xtarget.shape[0]

            X = np.vstack((Xdecoy, Xtarget))
            y = np.zeros((X.shape[0],))
            y[n_decoy:] = 1
            learner = sklearn.lda.LDA()
            model = learner.fit(X, y)
            w = model.scalings_.flatten()

            clf_scores = np.dot(train_df.get_full_feature_matrix_with_main_score(), w)
            clf_scores -= np.mean(clf_scores)
            train_df.add_classifier_score(clf_scores)
            train_df.df.sort(("is_decoy", "tg_id", "classifier_score"),
                    ascending=(False, True, False), inplace=True)

        # nach semi supervsised iter: classfiy hole dataset
        clf_scores = np.dot(exp.get_full_feature_matrix_with_main_score(), w)
        clf_scores -= np.mean(clf_scores)
        exp.add_classifier_score(clf_scores)
        exp.df.sort(("is_decoy", "tg_id", "classifier_score"),
                ascending=(False, True, False), inplace=True)

        top_decoy_indices, top_decoy_scores = exp.get_top_classifier_scores_decoy()
        mu = (np.mean(top_decoy_scores))
        nu = (np.std(top_decoy_scores, ddof=1))
        print "n=", len(top_decoy_scores), "mu=", mu, "nu=", nu

        classifier_scores = exp.df.classifier_score
        classifier_scores = (classifier_scores - mu)/nu
        exp.df.classifier_score = classifier_scores
        print len(classifier_scores), np.mean(classifier_scores), np.std(classifier_scores, ddof=1)

        # bis hierheir sit exp richtig gescored

        top_decoy_indices, top_decoy_scores = exp.get_top_classifier_scores_decoy()
        top_target_indices, top_target_scores = exp.get_top_classifier_scores_target()

        all_top_scores = list(top_decoy_scores) + list(top_target_scores)
        print len(all_top_scores), np.mean(all_top_scores), np.std(all_top_scores, ddof=1)

        # -> t.df_top_clfd_pt$d_score

        allix = top_decoy_indices + top_target_indices

        test = exp.df.loc[allix].test.values

        cdf = pd.DataFrame(dict(ds = all_top_scores,
                                class_ = [0] * len(top_decoy_scores) + [1]*len(top_target_scores),
                                test = test))

        collected.append(cdf)
        ws.append(w.flatten())


    w_final = np.vstack(ws).mean(axis=0)
    d_score = np.dot(exp.get_full_feature_matrix_with_main_score(), w_final)
    d_score -= np.mean(d_score)
    exp.add_classifier_score(d_score)

    top_decoy_indices, top_decoy_scores = exp.get_top_classifier_scores_decoy()
    top_target_indices, top_target_scores = exp.get_top_classifier_scores_target()

    mu = np.mean(top_decoy_scores)
    nu = np.std(top_decoy_scores, ddof=1)

    d_scores = [ (c-mu)/nu for c in d_score]
    exp.add_classifier_score(d_scores)
    top_target_indices, top_target_scores = exp.get_top_classifier_scores_target()

    error_stat_scores = []
    error_stat_labels = []
    for c in collected:
        error_stat_scores.extend(c[c.test].ds.values)
        error_stat_labels.extend(c[c.test].class_.values)


    good_scores = [s for (s,c) in  zip(error_stat_scores, error_stat_labels) if
            c]
    bad_scores = [s for (s,c) in  zip(error_stat_scores, error_stat_labels) if not c]

    errstat = pyprophet.stats.get_error_stat_from_null(good_scores, bad_scores)

    df_error = errstat.df
    num_null = errstat.num_null
    num_total = errstat.num
    summed_test_fraction_null = num_null / num_total

    # transfer statistics from test set to full set:
    num_top_target = len(top_target_scores)
    num_null_top_target = num_top_target * summed_test_fraction_null

    df_raw_stat = pyprophet.stats.get_error_table_using_percentile_positives_new(df_error,
                                                        top_target_scores,
                                                        num_null_top_target)

    q_values = pyprophet.stats.lookup_q_values_from_error_table(exp.df.classifier_score, df_raw_stat)
    exp.df["q_value"] = q_values

    final_err_table = pyprophet.stats.final_err_table(df_raw_stat)
    summary_err_table = pyprophet.stats.summary_err_table(df_raw_stat)

    print summary_err_table


if __name__ == "__main__":
    main()

