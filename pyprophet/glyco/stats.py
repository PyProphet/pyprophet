import click
import pandas as pd
import numpy as np
from collections import namedtuple

from scipy.interpolate import LinearNDInterpolator
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from scipy.stats import rankdata

from ..pyprophet import Result
from ..stats import pnorm, pemp, pi0est #, lfdr, qvalue
# from pyprophet.stats import stat_metrics
from ..stats import find_nearest_matches
from ..stats import posterior_chromatogram_hypotheses_fast

from ..ipf import compute_model_fdr
from ..optimized import count_num_positives
from .report import save_report


DensityModel = namedtuple('DensityModel', ('model', 'evaluate'))


class ErrorStatisticsCalculator:
    def __init__(self, scored_table,
                 density_estimator='gmm',
                 grid_size=256,
                 parametric=False,
                 pfdr=False,
                 tric_chromprob=False,
                 **kwargs):
        self.grid_size = grid_size
        self.parametric = parametric
        self.pi0_args = {
            (k if k == 'pi0_method' \
             else ('lambda_' if k == 'pi0_lambda' else k[len('pi0_'):])
            ): v
            for k, v in kwargs.items()
            if k.startswith('pi0_')
        }
        self.lfdr_args = {
            ('trunc' if k == 'lfdr_truncate' \
             else ('transf' if k == 'lfdr_transformation' else k[len('lfdr_'):])
            ): v
            for k, v in kwargs.items()
            if k.startswith('lfdr_')
        }
        self.pfdr = pfdr
        self.tric_chromprob = tric_chromprob

        self.scored_table = scored_table

        if density_estimator == 'kde':
            def density_model_func(scores, decoy_peptide=0, decoy_glycan=0):
                model = gaussian_kde(scores.T)
                return DensityModel(
                    model=model,
                    evaluate=lambda scores: model.evaluate(scores.T)
                )
        elif density_estimator == 'gmm':
            def density_model_func(scores, decoy_peptide=0, decoy_glycan=0):
                if decoy_peptide and decoy_glycan:
                    n_components = 1
                elif not decoy_peptide and not decoy_glycan:
                    n_components = 4
                else:
                    n_components = 2
                model = GaussianMixture(n_components=n_components)
                model.fit(scores)
                return DensityModel(
                    model=model,
                    evaluate=lambda scores: np.exp(model.score_samples(scores))
                )
        else:
            raise ValueError('invalid density_estimator: ' + str(density_estimator))
        self.density_model_func = density_model_func


    def get_top_scores(self, decoy_peptide, decoy_glycan,
                       score=['d_score_peptide', 'd_score_glycan', 'd_score_combined']):
        row = self.scored_table.apply(lambda x: True, axis=1)
        if 'peak_group_rank' in self.scored_table.columns:
            row &= self.scored_table['peak_group_rank'] == 1
        if decoy_peptide:
            row &= self.scored_table['decoy_peptide'] == 1
        elif decoy_peptide is not None:
            row &= self.scored_table['decoy_peptide'] == 0
        if decoy_glycan:
            row &= self.scored_table['decoy_glycan'] == 1
        elif decoy_glycan is not None:
            row &= self.scored_table['decoy_glycan'] == 0

        scores = self.scored_table.loc[row, score]
        scores = scores.dropna()
        return scores.values


    def calculate_partial_stat(self, part):
        pvalue = pnorm if self.parametric else pemp

        if part == 'both':
            score_part = 'd_score_combined'
            scores_target = self.get_top_scores(
                decoy_peptide=0,
                decoy_glycan=0,
                score=score_part
            )
            scores_decoy = self.get_top_scores(
                decoy_peptide=1,
                decoy_glycan=1,
                score=score_part
            )
        elif part == 'peptide':
            score_part = 'd_score_' + part
            scores_target = self.get_top_scores(
                decoy_peptide=0,
                decoy_glycan=None,
                score=score_part
            )
            scores_decoy = self.get_top_scores(
                decoy_peptide=1,
                decoy_glycan=None,
                score=score_part
            )
        elif part == 'glycan':
            score_part = 'd_score_' + part
            scores_target = self.get_top_scores(
                decoy_peptide=None,
                decoy_glycan=0,
                score=score_part
            )
            scores_decoy = self.get_top_scores(
                decoy_peptide=None,
                decoy_glycan=1,
                score=score_part
            )
        else:
            raise click.ClickException("Unspecified scoring part selected.")

        p_value = pvalue(scores_target, scores_decoy)
        # print(part)
        pi0 = pi0est(p_value, **self.pi0_args)
         # pep = lfdr(p_value, pi0=pi0['pi0'], **self.lfdr_args)

        stat = pd.DataFrame(scores_target, columns=[score_part])
        stat['p_value_' + part] = p_value
        # stat['pep_' + part] = pep

        if not hasattr(self, 'stats'):
            self.stats = {}
        self.stats[part] = stat

        if not hasattr(self, 'pi0'):
            self.pi0 = {}
        self.pi0[part] = pi0

        return stat, pi0


    def fit_density_models(self):
        decoy_dict = {
            'target': (0, 0),
            'decoy_peptide': (1, 0),
            'decoy_glycan': (0, 1),
            'decoy_both': (1, 1)
        }

        density_models = {}

        for decoy_type, decoy_args in decoy_dict.items():
            scores = self.get_top_scores(
                decoy_peptide=decoy_args[0],
                decoy_glycan=decoy_args[1],
                score=['d_score_peptide', 'd_score_glycan']
            )

            model = self.density_model_func(
                scores,
                decoy_peptide=decoy_args[0],
                decoy_glycan=decoy_args[1]
            )

            density_models[decoy_type] = model

        self.density_models = density_models

        return density_models


    def create_score_grid(self):
        def get_nonoutlier_range(x, lower=True, upper=False):
            q3, q1 = np.percentile(x, [75, 25])
            iqr = q3 - q1
            min_ = np.min(x)
            max_ = np.max(x)
            if lower:
                min_ = np.max([min_, q1 - 1.5 * iqr])
            if upper:
                max_ = np.min([max_, q3 + 1.5 * iqr])
            return (min_, max_)

        def get_grid_cutoffs(total_range, nonoutlier_range, num_cutoffs):
            margin = (nonoutlier_range[1] - nonoutlier_range[0]) * 0.05
            if total_range[0] < nonoutlier_range[0] and \
                total_range[1] > nonoutlier_range[1]:
                cutoffs = np.concatenate((
                    [total_range[0] - margin],
                    np.linspace(
                        nonoutlier_range[0] - margin,
                        nonoutlier_range[1] + margin,
                        num_cutoffs - 2
                    ),
                    [total_range[1] + margin]
                ))
            elif total_range[0] < nonoutlier_range[0]:
                cutoffs = np.concatenate((
                    [total_range[0] - margin],
                    np.linspace(
                        nonoutlier_range[0] - margin,
                        nonoutlier_range[1] + margin,
                        num_cutoffs - 1
                    )
                ))
            elif total_range[1] > nonoutlier_range[1]:
                cutoffs = np.concatenate((
                    np.linspace(
                        nonoutlier_range[0] - margin,
                        nonoutlier_range[1] + margin,
                        num_cutoffs - 1
                    ),
                    [total_range[1] + margin]
                ))
            else:
                cutoffs = np.linspace(
                    nonoutlier_range[0] - margin,
                    nonoutlier_range[1] + margin,
                    num_cutoffs - 1
                )
            return cutoffs

        decoy_dict = {
            'target': (0, 0),
            'decoy_peptide': (1, 0),
            'decoy_glycan': (0, 1),
            'decoy_both': (1, 1)
        }
        scores = {
            decoy_type: self.get_top_scores(
                decoy_peptide=decoy_args[0],
                decoy_glycan=decoy_args[1],
                score=['d_score_peptide', 'd_score_glycan']
            )
            for decoy_type, decoy_args in decoy_dict.items()
        }
        nonoutlier_ranges = np.array([
            (get_nonoutlier_range(s[:, 0]), get_nonoutlier_range(s[:, 1]))
            for decoy_type, s in scores.items()
        ])

        x_cutoffs = get_grid_cutoffs(
            nonoutlier_range=(
                np.min(nonoutlier_ranges[:, 0, 0]),
                np.max(nonoutlier_ranges[:, 0, 1])
            ),
            total_range=(
                self.scored_table['d_score_peptide'].min(),
                self.scored_table['d_score_peptide'].max()
            ),
            num_cutoffs=self.grid_size[0] \
                if isinstance(self.grid_size, tuple) or \
                    isinstance(self.grid_size, list) \
                else self.grid_size
        )
        y_cutoffs = get_grid_cutoffs(
            nonoutlier_range=(
                np.min(nonoutlier_ranges[:, 1, 0]),
                np.max(nonoutlier_ranges[:, 1, 1])
            ),
            total_range=(
                self.scored_table['d_score_glycan'].min(),
                self.scored_table['d_score_glycan'].max()
            ),
            num_cutoffs=self.grid_size[1] \
                if isinstance(self.grid_size, tuple) or \
                    isinstance(self.grid_size, list) \
                else self.grid_size
        )
        return np.meshgrid(x_cutoffs, y_cutoffs)


    def calculate_total_stat(self):
        pi00 = self.pi0['both']['pi0']
        pi01 = self.pi0['peptide']['pi0'] - pi00
        pi10 = self.pi0['glycan']['pi0'] - pi00
        pi11 = 1 - pi01 - pi10 - pi00

        if pi01 < 0:
            # raise ValueError('peptide pi0 < both pi0')
            pi01 = 0
        if pi10 < 0:
            # raise ValueError('glycan pi0 < both pi0')
            pi10 = 0
        if pi11 < 0:
            # raise ValueError('peptide pi0 + glycan pi0 - both pi0 > 1')
            pi11 = 0

        if not hasattr(self, 'density_models'):
            self.fit_density_models()

        X, Y = self.create_score_grid()
        scores = np.column_stack((X.ravel(), Y.ravel()))
        stat = pd.DataFrame(
            scores,
            columns=['d_score_peptide', 'd_score_glycan']
        )
        for decoy_type, model in self.density_models.items():
            stat['density_' + decoy_type] = model.evaluate(scores)

        stat['density_peptide_null_glycan_nonnull'] = np.maximum(
            (stat['density_decoy_peptide'].values - \
             (pi00 + pi10) * stat['density_decoy_both'].values) \
            / (pi01 + pi11), 0
        )
        stat['density_peptide_nonnull_glycan_null'] = np.maximum(
            (stat['density_decoy_glycan'].values - \
             (pi00 + pi01) * stat['density_decoy_both'].values) \
            / (pi10 + pi11), 0
        )
        stat['density_nonnull'] = np.maximum(
            (stat['density_target'].values - \
             pi00 * stat['density_decoy_both'].values - \
             pi01 * stat['density_peptide_null_glycan_nonnull'].values - \
             pi10 * stat['density_peptide_nonnull_glycan_null'].values) \
            / pi11, 0
        )

        pep_peptide = np.divide(
            pi00 * stat['density_decoy_both'].values + \
                pi01 * stat['density_peptide_null_glycan_nonnull'].values,
            stat['density_target'].values,
            out=np.zeros_like(stat['density_target'].values),
            where=stat['density_target'].values != 0
        )
        pep_glycan = np.divide(
            pi00 * stat['density_decoy_both'].values + \
                pi10 * stat['density_peptide_nonnull_glycan_null'].values,
            stat['density_target'].values,
            out=np.zeros_like(stat['density_target'].values),
            where=stat['density_target'].values != 0
        )
        pep_both = np.divide(
            pi00 * stat['density_decoy_both'].values,
            stat['density_target'].values,
            out=np.zeros_like(stat['density_target'].values),
            where=stat['density_target'].values != 0
        )
        pep_total = pep_peptide + pep_glycan - pep_both

        if self.lfdr_args.get('trunc', True):
            pep_peptide = np.minimum(pep_peptide, 1)
            pep_glycan = np.minimum(pep_glycan, 1)
            pep_both = np.minimum(pep_both, 1)
            pep_total = np.minimum(pep_total, 1)

        def monotonize(values, i0, j0, i_ascending=False, j_ascending=False):
            if i_ascending:
                for j in range(values.shape[1]):
                    for i in range(i0, 0, -1):
                        if values[i - 1, j] > values[i, j]:
                            values[i - 1, j] = values[i, j]
                    for i in range(i0, values.shape[0] - 1):
                        if values[i + 1, j] < values[i, j]:
                            values[i + 1, j] = values[i, j]
            elif i_ascending is not None:
                for j in range(values.shape[1]):
                    for i in range(i0, 0, -1):
                        if values[i - 1, j] < values[i, j]:
                            values[i - 1, j] = values[i, j]
                    for i in range(i0, values.shape[0] - 1):
                        if values[i + 1, j] > values[i, j]:
                            values[i + 1, j] = values[i, j]
            if j_ascending:
                for i in range(values.shape[0]):
                    for j in range(j0, 0, -1):
                        if values[i, j - 1] > values[i, j]:
                            values[i, j - 1] = values[i, j]
                    for i in range(j0, values.shape[1] - 1):
                        if values[i, j + 1] < values[i, j]:
                            values[i, j + 1] = values[i, j]
            elif j_ascending is not None:
                for i in range(values.shape[0]):
                    for j in range(j0, 0, -1):
                        if values[i, j - 1] < values[i, j]:
                            values[i, j - 1] = values[i, j]
                    for j in range(j0, values.shape[1] - 1):
                        if values[i, j + 1] > values[i, j]:
                            values[i, j + 1] = values[i, j]
            return values

        if self.lfdr_args.get('monotone', True):
            i0, j0 = np.unravel_index(
                np.argmax(stat['density_nonnull'].values),
                X.shape
            )
            monotonize(pep_peptide.reshape(X.shape), i0=i0, j0=j0)
            monotonize(pep_glycan.reshape(X.shape), i0=i0, j0=j0)
            monotonize(pep_both.reshape(X.shape), i0=i0, j0=j0)
            monotonize(pep_total.reshape(X.shape), i0=i0, j0=j0)

        stat['pep_both'] = pep_both
        stat['pep_peptide'] = pep_peptide
        stat['pep_glycan'] = pep_glycan
        stat['pep'] = pep_total

        def qvalue(pep, density, X, Y):
            dX = np.diff(X, axis=1)
            dX = np.column_stack((dX, np.min(dX, axis=1)))
            dY = np.diff(Y, axis=0)
            dY = np.row_stack((dY, np.min(dY, axis=0)))
            dS = dX * dY
            ds = dS.ravel()

            pep = pep.ravel()
            density = density.ravel()
            order = np.argsort(pep)
            frac_fp = np.cumsum(pep[order] * density[order] * ds[order])
            frac_positive = np.cumsum(density[order] * ds[order])
            fdr_ordered = np.divide(
                frac_fp,
                frac_positive,
                out=np.zeros_like(pep),
                where=frac_positive != 0
            )
            ranks = rankdata(pep, method='max')
            fdr = fdr_ordered[ranks - 1]
            return fdr

        stat['q_value'] = qvalue(
            stat['pep'].values,
            stat['density_target'].values,
            X, Y
        )

        if not hasattr(self, 'stats'):
            self.stats = {}
        self.stats['total'] = stat

        return stat


    def lookup_stat_table(self, scored_table, stat, by, value=None):
        if value is None:
            value = [
                col for col in stat.columns
                if col not in scored_table.columns
            ]
        if isinstance(value, str):
            value = [value]

        row_na = scored_table[by].isna()
        idx = find_nearest_matches(
            np.float32(stat[by].values),
            np.float32(scored_table[by].loc[~row_na].values)
        )

        for col in value:
            scored_table[col] = scored_table[by].map(lambda x: np.nan)
            scored_table.loc[~row_na, col] = \
                stat[col].iloc[idx].values

#        stat = stat.drop_duplicates(subset=by).sort_values(by=by)
#        for col in value:
#            f = np.interp(
#                scored_table[by].loc[~row_na].values,
#                stat[by].values, stat[col].values,
#            )
#            scored_table[col] = scored_table[by].map(lambda x: np.nan)
#            scored_table.loc[~row_na, col] = f

        return scored_table


    def lookup_stat_grid(self, scored_table, stat,
                         by_x='d_score_peptide', by_y='d_score_glycan',
                         value=None):
        if value is None:
            value = [
                col for col in stat.columns
                if col not in scored_table.columns
            ]
        if isinstance(value, str):
            value = [value]

        row_na = scored_table[by_x].isna() | \
            scored_table[by_y].isna()
        for col in value:
            ip = LinearNDInterpolator(
                points=stat[[by_x, by_y]].values,
                values=stat[col].values
            )
            f = ip(
                scored_table.loc[~row_na, [by_x, by_y]].values
            )
            scored_table[col] = scored_table[by_x] \
                .map(lambda x: np.nan)
            scored_table.loc[~row_na, col] = f
        return scored_table


    def lookup_error_stat(self, scored_table, stats):
        for part in ['peptide', 'glycan', 'both']:
            if part == 'both':
                score_part = 'd_score_combined'
            else:
                score_part = 'd_score_' + part
            stat = stats.get(part, None)
            if stat is not None:
                scored_table = self.lookup_stat_table(
                    scored_table, stat,
                    by=score_part
                )

        stat = stats.get('total', None)
        if stat is not None:
            scored_table = self.lookup_stat_grid(
                scored_table, stat,
                value=['pep_peptide', 'pep_glycan', 'pep_both', 'pep']
            )

        return scored_table


    def qvalue(self, scored_table):
        target = (scored_table['decoy_peptide'] == 0) & \
            (scored_table['decoy_glycan'] == 0)
        if 'peak_group_rank' in scored_table.columns:
            target &= scored_table['peak_group_rank'] == 1

        qvalue = scored_table['pep'].map(lambda x: 0.0)
        qvalue.loc[~target] = self.lookup_stat_grid(
            scored_table.loc[~target].copy(), self.stats['total'],
            value=['q_value']
        )['q_value']

        qvalue.loc[target] = compute_model_fdr(scored_table.loc[target, 'pep'])

        scored_table['q_value'] = qvalue
        return scored_table


    def chromatogram_probabilities(self, scored_table):
        pi0 = self.pi0['peptide']['pi0'] + self.pi0['glycan']['pi0'] - \
            self.pi0['both']['pi0']

        texp = namedtuple('DummyExperiment', ['df'])(
            df = scored_table[['group_id', 'precursor_id', 'pep']] \
                .rename(columns={
                    'precursor_id': 'tg_num_id'
                })
        )
        allhypothesis, h0 = posterior_chromatogram_hypotheses_fast(
            texp, pi0
        )
        scored_table['h_score'] = allhypothesis
        scored_table['h0_score'] = h0
        return scored_table


    def calculate_stat_metrics(self):
        pi0_both = self.pi0['both']['pi0']
        pi0_peptide = self.pi0['peptide']['pi0']
        pi0_glycan = self.pi0['glycan']['pi0']
        pi0_total = pi0_peptide + pi0_glycan - pi0_both

#        decoy_dict = {
#            'target': (0, 0),
#            'decoy_peptide': (1, 0),
#            'decoy_glycan': (0, 1),
#            'decoy_both': (1, 1)
#        }
#        scores = {
#            decoy_type: -self.get_top_scores(
#                decoy_peptide=decoy_args[0],
#                decoy_glycan=decoy_args[1],
#                score='pep'
#            )
#            for decoy_type, decoy_args in decoy_dict.items()
#        }
#
#        pvalue = pnorm if self.parametric else pemp
#        fpr_both = pvalue(scores['target'], scores['decoy_both'])
#        fpr_peptide = pvalue(scores['target'], scores['decoy_peptide'])
#        fpr_glycan = pvalue(scores['target'], scores['decoy_glycan'])
#        fpr = (fpr_peptide * pi0_peptide + fpr_glycan * pi0_glycan - \
#            fpr_both * pi0_both) / pi0_total
#
#        stat = pd.DataFrame(
#            np.column_stack((-scores['target'], fpr)),
#            columns=['pep', 'p_value']
#        )
#        stat['q_value'] = compute_model_fdr(stat['pep'].values)
#        stat.sort_values(by='pep', ascending=False, inplace=True)
#        stat.reset_index(drop=True, inplace=True)
#        metrics = stat_metrics(stat['p_value'].values, pi0=pi0_total, pfdr=self.pfdr)
#        stat = pd.concat(
#            (stat, metrics),
#            axis=1
#        )

        stat = pd.DataFrame(
            self.get_top_scores(
                decoy_peptide=0,
                decoy_glycan=0,
                score=['pep', 'q_value']
            ),
            columns=['pep', 'q_value']
        )
        stat.sort_values(by='pep', ascending=False, inplace=True)
        stat.reset_index(drop=True, inplace=True)

        num_total = len(stat['pep'].values)
        num_positives = count_num_positives(stat['pep'].values)
        num_negatives = num_total - num_positives
        num_null = pi0_total * num_total

        fp = num_positives * stat['q_value'].values
        tp = num_positives - fp
        fpr = np.minimum(fp / num_null, 1.0)
        tn = num_null * (1.0 - fpr)
        fn = num_negatives - num_null * (1.0 - fpr)

        fdr = np.divide(fp, num_positives, out=np.zeros_like(fp), where=num_positives!=0)
        fnr = np.divide(fn, num_negatives, out=np.zeros_like(fn), where=num_negatives!=0)

        if self.pfdr:
            fdr /= (1.0 - (1.0 - fpr) ** num_total)
            fdr[fpr == 0] = 1.0 / num_total

            fnr /= 1.0 - fpr ** num_total
            fnr[fpr == 0] = 1.0 / num_total

        sens = tp / (num_total - num_null)
        sens[sens < 0.0] = 0.0
        sens[sens > 1.0] = 1.0

        fdr[fdr < 0.0] = 0.0
        fdr[fdr > 1.0] = 1.0
        fdr[num_positives == 0] = 0.0

        fnr[fnr < 0.0] = 0.0
        fnr[fnr > 1.0] = 1.0
        fnr[num_positives == 0] = 0.0

        svalues = pd.Series(sens)[::-1].cummax()[::-1]
        metrics = pd.DataFrame({
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'fpr': fpr, 'fdr': fdr, 'fnr': fnr,
            'svalue': svalues
        })
        stat = pd.concat(
            (stat, metrics),
            axis=1
        )

        self.stat_metrics = stat
        return stat


    def final_error_stat(self):
        stat = self.lookup_stat_table(
            self.stats['total'].copy(),
            self.stat_metrics,
            by='pep'
        )
        return stat


    def summary_error_stat(self,
                           qvalues=[0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        qvalues = np.array(qvalues)
        idx = find_nearest_matches(
            np.float32(self.stat_metrics['q_value'].values),
            np.float32(qvalues)
        )

        stat_sub = self.stat_metrics.iloc[idx].copy()
        for i_sub, (i0, i1) in enumerate(zip(idx, idx[1:])):
            if i1 == i0:
                stat_sub.iloc[i_sub + 1, :] = None
        stat_sub['q_value'] = qvalues
        stat_sub.reset_index(inplace=True, drop=True)

        summary_stat = stat_sub[[
            'q_value', 'pep',
            'svalue',
            'fdr', 'fnr', 'fpr',
            'tp', 'tn', 'fp', 'fn'
        ]]
        return summary_stat


    def error_statistics(self):
        for part in ['peptide', 'glycan', 'both']:
            self.calculate_partial_stat(part)

        self.calculate_total_stat()

        self.scored_table = self.lookup_error_stat(self.scored_table, self.stats)
        self.scored_table = self.qvalue(self.scored_table)

        if self.tric_chromprob:
            self.scored_table = \
                self.chromatogram_probabilities(self.scored_table)

        self.calculate_stat_metrics()
        final_statistics =  self.final_error_stat()
        summary_statistics = self.summary_error_stat()

        # result = {
        #     'scored_table': self.scored_table,
        #     'final_statistics': final_statistics,
        #     'summary_statistics': summary_statistics
        # }
        scored_table = self.scored_table
        result = Result(summary_statistics, final_statistics, scored_table)
        
        pi0 = self.pi0

        return result, pi0
    
def statistics_report(data, outfile, context, analyte, 
                      **kwargs):
    '''
    Adapted from https://github.com/lmsac/GproDIA/blob/main/src/glycoprophet/level_contexts.py
    '''

    error_stat = ErrorStatisticsCalculator(data, **kwargs)    
    result, pi0 = error_stat.error_statistics()
    
    # print summary table
    summary = result.summary_statistics
    if summary is not None:
        summary = summary.get('total', None)
    
    if summary is not None:
        click.echo("=" * 80)
        click.echo(summary)
        click.echo("=" * 80)

    if context == 'run-specific':
        outfile = outfile + "_" + str(data['run_id'].unique()[0])

    # export PDF report
    save_report(
        outfile + "_" + context + "_" + analyte + ".pdf", 
        outfile + ": \n" + context + " " + analyte + "-level error-rate control", 
        result.scored_tables,
        result.final_statistics,
        pi0
    )
    return result.scored_tables