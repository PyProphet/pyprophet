try:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d as plt3d

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import cm
except ImportError:
    plt = None
    plt3d = None

import click
import numpy as np
from scipy.stats import gaussian_kde


def transparent_cmap(cmap, alpha=(0.1, 0.7, 0.75, 1.0)):
    x = np.linspace(0.0, 1.0, cmap.N)
    y = cmap(x)

    if isinstance(alpha, float) or isinstance(alpha, int):
        a = alpha
    elif len(alpha) == 1:
        a = alpha[0]
    else:
        a = np.zeros(cmap.N)
        n = np.linspace(0, cmap.N - 1, len(alpha), dtype=int)
        for i in range(len(n) - 1):
            a[n[i]:(n[i + 1] + 1)] = np.linspace(
                alpha[i], alpha[i + 1],
                n[i + 1] - n[i] + 1
            )

    y[:, 3] *= a
    return LinearSegmentedColormap.from_list(
        cmap.name + '_transperent',
        y, cmap.N
    )


def get_grouped_scores(scored_table, score):
    if 'peak_group_rank' in scored_table.columns:
        scored_table = scored_table.loc[scored_table['peak_group_rank'] == 1]

    if 'decoy_peptide' in scored_table.columns and \
        'decoy_glycan' in scored_table.columns:
        decoy_dict = {
            'target': (0, 0),
            'decoy_peptide': (1, 0),
            'decoy_glycan': (0, 1),
            'decoy_both': (1, 1)
        }
        scores = {
            decoy_type: scored_table.loc[
                (scored_table['decoy_peptide'] == decoy_args[0]) & \
                (scored_table['decoy_glycan'] == decoy_args[1]),
                score
            ].values
            for decoy_type, decoy_args in decoy_dict.items()
        }
    elif 'decoy' in scored_table.columns:
        decoy_dict = {
            'target': 0,
            'decoy': 1
        }
        scores = {
            decoy_type: scored_table.loc[
                (scored_table['decoy'] == decoy_args),
                score
            ].values
            for decoy_type, decoy_args in decoy_dict.items()
        }
    else:
        raise ValueError('no decoy column')
    return scores


def get_score_ranges(scores, exclude_outlier='lower'):
    def get_nonoutlier_range(x, lower, upper):
        q3, q1 = np.percentile(x, [75, 25])
        iqr = q3 - q1
        min_ = np.min(x)
        max_ = np.max(x)
        if lower:
            min_ = np.max([min_, q1 - 1.5 * iqr])
        if upper:
            max_ = np.min([max_, q3 + 1.5 * iqr])
        return (min_, max_)

    if not exclude_outlier or exclude_outlier == 'none':
        lower = False
        upper = False
    elif exclude_outlier == 'lower':
        lower = True
        upper = False
    elif exclude_outlier == 'upper':
        lower = False
        upper = True
    elif exclude_outlier == 'both' or exclude_outlier == True:
        lower = True
        upper = True
    else:
        raise ValueError('invalid exclude_outlier: ' + str(exclude_outlier))

    nonoutlier_ranges = np.array([
        np.apply_along_axis(
            get_nonoutlier_range, axis=0, arr=s,
            lower=lower, upper=upper
        )
        for decoy_type, s in scores.items()
    ])
    if len(nonoutlier_ranges.shape) == 2:
        return (np.min(nonoutlier_ranges[:, 0]),
                np.max(nonoutlier_ranges[:, 1]))
    else:
        return (np.min(nonoutlier_ranges[:, 0, :], axis=0),
                np.max(nonoutlier_ranges[:, 1, :], axis=0))


def plot_score_hist(ax, scored_table, score,
                    title=None, xlabel=None, ylabel=None,
                    legend=True, exclude_outlier='lower',
                    **kwargs):
    if title is None:
        title = score + ' distributions'
    if xlabel is None:
        xlabel = score
    if ylabel is None:
        ylabel = '# of groups'
    if not isinstance(score, str):
        raise TypeError('invalid score: ' + str(score))

    scores = get_grouped_scores(scored_table, score)
    x_range = get_score_ranges(scores, exclude_outlier=exclude_outlier)

    decoy_type = ['target', 'decoy_peptide', 'decoy_glycan', 'decoy_both']
    if all((d in scores for d in decoy_type)):
        color = ['g', 'y', 'b', 'r']
    else:
        decoy_type = ['target', 'decoy']
        color = ['g', 'r']

    kwargs.setdefault('bins', 20)
    kwargs.setdefault('color', color)
    kwargs.setdefault(
        'label',
        [d.replace('_', ' ').capitalize() for d in decoy_type]
    )
    kwargs.setdefault('histtype', 'bar')

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_range[0], x_range[1])
    ax.hist(
        [scores[d] for d in decoy_type],
        **kwargs
    )
    if isinstance(legend, dict):
        ax.legend(**legend)
    elif legend:
        ax.legend()

    return ax


def plot_score_density(ax, scored_table, score,
                       title=None, xlabel=None, ylabel=None,
                       legend=True, exclude_outlier='lower',
                       **kwargs):
    if title is None:
        title = score + ' distributions'
    if xlabel is None:
        xlabel = score
    if ylabel is None:
        ylabel = 'Density'
    if not isinstance(score, str):
        raise TypeError('invalid score: ' + str(score))

    scores = get_grouped_scores(scored_table, score)
    x_range = get_score_ranges(scores, exclude_outlier=exclude_outlier)

    def get_density(scores, cutoffs):
        model = gaussian_kde(scores)
        model.covariance_factor = lambda: .25
        model._compute_covariance()
        return model(cutoffs)

    x_cutoffs = np.linspace(x_range[0], x_range[1], 200)
    density = {
        decoy_type: get_density(values, x_cutoffs)
        for decoy_type, values in scores.items()
    }

    decoy_type = ['target', 'decoy_peptide', 'decoy_glycan', 'decoy_both']
    if all((d in scores for d in decoy_type)):
        color = ['g', 'y', 'b', 'r']
    else:
        decoy_type = ['target', 'decoy']
        color = ['g', 'r']

    kwargs.setdefault('color', color)
    kwargs.setdefault(
        'label',
        [d.replace('_', ' ').capitalize() for d in decoy_type]
    )

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_range[0], x_range[1])
    for i, d in enumerate(decoy_type):
        ax.plot(
            x_cutoffs, density[d],
            **({k: v[i] if isinstance(v, list) else v
                for k, v in kwargs.items()})
        )

    if isinstance(legend, dict):
        ax.legend(**legend)
    elif legend:
        ax.legend()
    return ax


def plot_dscore_scatter(ax, scored_table, max_num=1000,
                        title=None, xlabel=None, ylabel=None,
                        legend=False, exclude_outlier='lower',
                        **kwargs):
    if title is None:
        title = 'Peptide/Glycan D-score'
    if xlabel is None:
        xlabel = 'Peptide D-score'
    if ylabel is None:
        ylabel = 'Glycan D-score'

    scores = get_grouped_scores(
        scored_table,
        score=['d_score_peptide', 'd_score_glycan']
    )

    ranges = get_score_ranges(scores, exclude_outlier=exclude_outlier)
    x_range = (ranges[0][0], ranges[1][0])
    y_range = (ranges[0][1], ranges[1][1])

    resampled = False
    for decoy_type in scores:
        if scores[decoy_type].shape[0] > max_num:
            scores[decoy_type] = scores[decoy_type] \
                [np.random.choice(scores[decoy_type].shape[0], max_num), :]
            resampled = True

    if resampled:
        title += ' (resampled each group <= %s)' % max_num

    decoy_type = ['target', 'decoy_peptide', 'decoy_glycan', 'decoy_both']
    kwargs.setdefault('color', ['g', 'y', 'b', 'r'])
    kwargs.setdefault(
        'label',
        [d.replace('_', ' ').capitalize() for d in decoy_type]
    )
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('alpha', 0.25)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    for i, d in enumerate(decoy_type):
        ax.scatter(
            scores[d][:, 0], scores[d][:, 1],
            **({k: v[i] if isinstance(v, list) else v
                for k, v in kwargs.items()})
        )

    if isinstance(legend, dict):
        ax.legend(**legend)
    elif legend:
        ax.legend()
    return ax


def plot_contour(ax, final_statistics, value,
                 levels=10, fontsize=10,
                 title=None, xlabel=None, ylabel=None,
                 legend=False,
                 **kwargs):
    if isinstance(value, str):
        if title is None:
            title = value
        value = [value]
        for k in kwargs:
            if isinstance(kwargs[k], list):
                kwargs[k] = [kwargs[k]]
    elif isinstance(value, list):
        if title is None:
            title = str(value)
    else:
        raise TypeError('invalid value: ' + str(value))

    if xlabel is None:
        xlabel = 'Peptide D-score'
    if ylabel is None:
        ylabel = 'Glycan D-score'

    x = np.sort(final_statistics['d_score_peptide'].unique())
    y = np.sort(final_statistics['d_score_glycan'].unique())
    X, Y = np.meshgrid(x, y)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for i, v in enumerate(value):
        Z = final_statistics \
            .pivot_table(
                index='d_score_peptide',
                columns='d_score_glycan',
                values=v
            ).values.T
        c = ax.contour(
            X, Y, Z, levels,
            **({k: v[i] if isinstance(v, list) else v
                for k, v in kwargs.items()})
        )
        ax.clabel(c, inline=True, fontsize=fontsize)

    if isinstance(legend, dict):
        ax.legend(**legend)
    elif legend:
        ax.legend()
    return ax


def plot_3d_surface(ax, final_statistics, value,
                    title=None, xlabel=None, ylabel=None, zlabel=None,
                    legend=False,
                    **kwargs):
    if isinstance(value, str):
        if title is None:
            title = value
        if zlabel is None:
            zlabel = value
        value = [value]
        for k in kwargs:
            if isinstance(kwargs[k], list):
                kwargs[k] = [kwargs[k]]
    elif isinstance(value, list):
        if title is None:
            title = str(value)
        if zlabel is None:
            zlabel = str(value)
    else:
        raise TypeError('invalid value: ' + str(value))

    if xlabel is None:
        xlabel = 'Peptide D-score'
    if ylabel is None:
        ylabel = 'Glycan D-score'

    x = np.sort(final_statistics['d_score_peptide'].unique())
    y = np.sort(final_statistics['d_score_glycan'].unique())
    X, Y = np.meshgrid(x, y)

    if ax is None:
        fig = plt.figure()
        ax = plt3d.Axes3D(fig)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    for i, v in enumerate(value):
        Z = final_statistics \
            .pivot_table(
                index='d_score_peptide',
                columns='d_score_glycan',
                values=v,
                dropna=False,
            ).values.T
        c = ax.plot_surface(
            X, Y, Z,
            **({k: v[i] if isinstance(v, list) else v
                for k, v in kwargs.items()})
        )

    label = kwargs.get('label', None)
    if (legend or isinstance(legend, dict)) and label is not None:
        if not isinstance(label, list):
            label = [label] * len(value)

        color = kwargs.get('color', None)
        if color is None:
            def cmap_to_color(cmap):
                if isinstance(cmap, str):
                    cmap = cm.get_cmap(cmap)
                return cmap(0.75)

            cmap = kwargs.get('cmap', None)
            if isinstance(cmap, list):
                color = [cmap_to_color(c) for c in cmap]
            elif cmap is not None:
                color = cmap_to_color(cmap)
        if color is None:
            color = tuple(c._edgecolors3d[0])

        fake_line2d = []
        for i, v in enumerate(value):
            if isinstance(color, list):
                c = color[i]
            elif color is not None:
                c = color
            fake_line2d.append(
                matplotlib.lines.Line2D(
                    [0], [0],
                    linestyle='none',
                    c=c, marker='o'
                )
            )

        if isinstance(legend, dict):
            ax.legend(fake_line2d, label, **legend)
        elif legend:
            ax.legend(fake_line2d, label)

    return ax


def plot_pi0_hist(ax, scored_table, pi0, part,
                  title=None, xlabel=None, ylabel=None,
                  **kwargs):
    if part is not None:
        pvalues = get_grouped_scores(scored_table, score='p_value_' + part)

        if part == 'peptide':
            pvalues = np.concatenate((pvalues['target'], pvalues['decoy_glycan']))
        elif part == 'glycan':
            pvalues = np.concatenate((pvalues['target'], pvalues['decoy_peptide']))
        elif part == 'both':
            pvalues = pvalues['target']
        else:
            raise ValueError('invalid part: ' + str(part))

        pi0_ = pi0[part]

        if title is None:
            title = part.capitalize() + ' P-value density histogram: ' + \
                '$\pi_0$ = ' + str(np.around(pi0_['pi0'], decimals=3))
        if xlabel is None:
            xlabel = part.capitalize() + ' P-value'
    else:
        pvalues = get_grouped_scores(scored_table, score='p_value')['target']
        pi0_ = pi0
        if title is None:
            title = 'P-value density histogram: ' + \
                '$\pi_0$ = ' + str(np.around(pi0_['pi0'], decimals=3))
        if xlabel is None:
            xlabel = 'P-value'

    if ylabel is None:
        ylabel = 'Density'

    kwargs.setdefault('bins', 20)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.hist(pvalues, density=True, **kwargs)
    ax.plot([0, 1], [pi0_['pi0'], pi0_['pi0']], 'r')

    return ax


def plot_pi0_smooth(ax, pi0, part,
                    title=None, xlabel=None, ylabel=None,
                    **kwargs):
    if part is not None:
        pi0_ = pi0[part]
        if title is None:
            title = part.capitalize() + ' $\pi_0$ smoothing fit plot'
        if ylabel is None:
            ylabel = part.capitalize() + ' $\pi_0$($\lambda$)'
    else:
        pi0_ = pi0
        if title is None:
            title = '$\pi_0$ smoothing fit plot'
        if ylabel is None:
            ylabel = '$\pi_0$($\lambda$)'

    if xlabel is None:
        xlabel = '$\lambda$'

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.plot(pi0_['lambda_'], pi0_['pi0_lambda'], '.')
    ax.plot(pi0_['lambda_'], pi0_['pi0_smooth'], 'r')

    return ax


def plot_stat_curves(ax, final_statistics,
                     value_x, value_y,
                     cutoff='pep', num_cutoffs=21,
                     title=None, xlabel=None, ylabel=None,
                     legend=False,
                     **kwargs):
    if not isinstance(value_x, str):
        raise TypeError('invalid value_x: ' + str(value_x))
    if isinstance(value_y, str):
        if title is None:
            title = value_y + '/' + value_x
        if ylabel is None:
            ylabel = value_y
        value_y = [value_y]
        for k in kwargs:
            if isinstance(kwargs[k], list):
                kwargs[k] = [kwargs[k]]
    elif isinstance(value_y, list):
        if title is None:
            title = str(value_y) + '/' + value_x
        if ylabel is None:
            ylabel = str(value_y)
    else:
        raise TypeError('invalid value_y: ' + str(value_y))

    final_statistics = final_statistics \
        .drop_duplicates(subset=cutoff) \
        .sort_values(by=cutoff)

    values = {
        k: final_statistics[k].values
        for k in set([value_x] + value_y)
    }

    if num_cutoffs is not None:
        t = final_statistics[cutoff].values
        tt = np.linspace(np.min(t), np.max(t), num_cutoffs)
        for k in values:
            values[k] = np.interp(tt, t, values[k])

    if xlabel is None:
        xlabel = value_x

    kwargs.setdefault('marker', 'o')
    kwargs.setdefault('markersize', 3)

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    for i, k in enumerate(value_y):
        ax.plot(
            values[value_x], values[k],
            **({k: v[i] if isinstance(v, list) else v
                for k, v in kwargs.items()})
        )

    if isinstance(legend, dict):
        ax.legend(**legend)
    elif legend:
        ax.legend()
    return ax


def save_report(pdf_path, title,
                scored_table,
                final_statistics,
                pi0):
    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        grid = plt.GridSpec(3, 2)
        ax = fig.add_subplot(grid[:-1, :])
        plot_dscore_scatter(ax, scored_table, legend=True)
        plot_contour(
            ax, final_statistics,
            value='q_value', title='Q-value',
            levels=[0.01, 0.02, 0.05, 0.1],
            colors='purple'
        )

        ax = fig.add_subplot(grid[-1, 0])
        plot_stat_curves(
            ax, final_statistics,
            value_x='q_value', value_y='svalue',
            title='Q-value/S-value',
            xlabel='False discovery rate (Q-value)',
            ylabel='True positive rate (S-value)'
        )
        ax = fig.add_subplot(grid[-1, 1])
        plot_stat_curves(
            ax, final_statistics,
            value_x='pep', value_y=['q_value', 'svalue'],
            title='Score performance',
            xlabel='Posterior error probability',
            ylabel='Rates',
            label=['Q-value', 'S-value'],
            color=['r', 'g'],
            legend=True
        )
        fig.text(0.5, 0.05, 'Points may be resampled to reduce data size', ha='center')
        if title is not None:
            fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)


        fig = plt.figure(figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5, wspace=0.25)
        grid = plt.GridSpec(3, 3)
        ax = fig.add_subplot(grid[:-1, :])
        plot_dscore_scatter(ax, scored_table)
        plot_contour(
            ax, final_statistics,
            value='pep', title='Posterior error probability',
            levels=[0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
            colors='purple',
            legend=True
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for i, part in enumerate(['both', 'peptide', 'glycan']):
            ax = fig.add_subplot(grid[-1, i])
            plot_dscore_scatter(ax, scored_table, max_num=100, s=5, alpha=0.75)
            plot_contour(
                ax, final_statistics,
                value='pep_' + part, title=part.capitalize() + ' PEP',
                levels=[0.01, 0.05, 0.1, 0.2, 0.4],
                colors='purple', linewidths=0.25, fontsize=4
            )
        fig.text(0.5, 0.05, 'Points may be resampled to reduce data size', ha='center')
        if title is not None:
            fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)


        if plt3d is None:
            click.echo("Warning: The mpl_toolkits.mplot3d package is required to create 3-D plots.")
        else:
            fig = plt.figure(figsize=(10, 15))
            fig.subplots_adjust(hspace=0.5)
            grid = plt.GridSpec(3, 2)
            cmaps = [cm.Greens, cm.YlOrBr, cm.Blues, cm.Reds]
            for i, density in enumerate([
                'density_target', 'density_decoy_peptide',
                'density_decoy_glycan', 'density_decoy_both'
            ]):
                ax = fig.add_subplot(grid[i], projection='3d')
                ax.set_xlim(xlim[0], xlim[1])
                ax.set_ylim(ylim[0], ylim[1])
                plot_3d_surface(
                    ax, final_statistics,
                    value=density,
                    title=density.replace('_', ' ').capitalize(),
                    zlabel='Density',
                    cmap=transparent_cmap(cmaps[i], (0, 0.9, 0.95, 1))
                )

            ax = fig.add_subplot(grid[-1, :], projection='3d')
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            plot_3d_surface(
                ax, final_statistics,
                value=[
                    'density_nonnull', 'density_peptide_null_glycan_nonnull',
                    'density_peptide_nonnull_glycan_null', 'density_decoy_both'
                ],
                label=['Non-null', 'Peptide null', 'Glycan null', 'Both null'],
                title='Four-group mixture density',
                zlabel='Density',
                legend=dict(loc=2),
                cmap=[transparent_cmap(cmap) for cmap in cmaps],
                zorder=[0.25, 0.0, 1.0, 0.5]
            )
            if title is not None:
                fig.suptitle(title)
            pdf.savefig(fig)
            plt.close(fig)


        fig = plt.figure(figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        for i, score_part in enumerate(['combined', 'peptide', 'glycan']):
            ax = fig.add_subplot(3, 2, 1 + 2 * i)
            plot_score_hist(
                ax, scored_table,
                score='d_score_' + score_part,
                title=score_part.capitalize() + ' D-score',
                xlabel=score_part.capitalize() + ' D-score'
            )
            ax = fig.add_subplot(3, 2, 2 + 2 * i)
            plot_score_density(
                ax, scored_table,
                score='d_score_' + score_part,
                title=score_part.capitalize() + ' D-score',
                xlabel=score_part.capitalize() + ' D-score'
            )
        if title is not None:
            fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)


        fig = plt.figure(figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        for i, part in enumerate(['both', 'peptide', 'glycan']):
            ax = fig.add_subplot(3, 2, 1 + 2 * i)
            plot_pi0_hist(ax, scored_table, pi0, part=part)
            if pi0[part]['pi0_smooth'] is not False:
                ax = fig.add_subplot(3, 2, 2 + 2 * i)
                plot_pi0_smooth(ax, pi0, part=part)
        fig.text(
            0.5, 0.925,
            'Total $\pi_0$ = ' + str(np.around(
                pi0['peptide']['pi0'] + pi0['glycan']['pi0'] - \
                    pi0['both']['pi0'],
                decimals=3
            )),
            ha='center',
            fontsize=12
        )
        if title is not None:
            fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)


def save_report_pyprophet(pdf_path, title,
                          scored_table,
                          final_statistics,
                          pi0):
    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(10, 15))
        fig.subplots_adjust(hspace=0.5)
        ax = fig.add_subplot(3, 2, 1)
        plot_stat_curves(
            ax, final_statistics, num_cutoffs=None,
            cutoff='cutoff',
            value_x='qvalue', value_y='svalue',
            title='Q-value/S-value',
            xlabel='False discovery rate (Q-value)',
            ylabel='True positive rate (S-value)'
        )
        ax = fig.add_subplot(3, 2, 2)
        plot_stat_curves(
            ax, final_statistics, num_cutoffs=None,
            cutoff='cutoff',
            value_x='cutoff', value_y=['qvalue', 'svalue'],
            title='Score performance',
            xlabel='D-score',
            ylabel='Rates',
            label=['Q-value', 'S-value'],
            color=['r', 'g'],
            legend=True
        )

        ax = fig.add_subplot(3, 2, 3)
        plot_score_hist(
            ax, scored_table,
            score='d_score',
            title='D-score',
            xlabel='D-score'
        )
        ax = fig.add_subplot(3, 2, 4)
        plot_score_density(
            ax, scored_table,
            score='d_score',
            title='D-score',
            xlabel='D-score'
        )

        ax = fig.add_subplot(3, 2, 5)
        plot_pi0_hist(ax, scored_table, pi0, part=None)
        if pi0['pi0_smooth'] is not False:
            ax = fig.add_subplot(3, 2, 6)
            plot_pi0_smooth(ax, pi0, part=None)

        if title is not None:
            fig.suptitle(title)
        pdf.savefig(fig)
        plt.close(fig)



def plot_scores(df, out, title=None):
    if plt is None:
        raise ImportError("Error: The matplotlib package is required to create a report.")

    df = df.rename(columns={
        x: x.lower()
        for x in df.columns \
            .intersection(['DECOY_PEPTIDE', 'DECOY_GLYCAN', 'DECOY'])
    })

    score_columns = df.columns.intersection(['SCORE']).tolist() + \
        df.columns.intersection(['SCORE_PEPTIDE', 'SCORE_GLYCAN']).tolist() + \
        df.columns[df.columns.str.startswith("MAIN_VAR_")].tolist() + \
        df.columns[df.columns.str.startswith("VAR_")].tolist()

    with PdfPages(out) as pdf:
        for score in score_columns:
            if df[score].isnull().values.any():
                continue

            fig = plt.figure(figsize=(10, 10))
            fig.subplots_adjust(hspace=0.5)
            ax = fig.add_subplot(2, 1, 1)
            plot_score_hist(
                ax, df,
                score=score,
                title=score,
                xlabel=score,
                legend=dict(loc=2),
                exclude_outlier=False
            )
            ax = fig.add_subplot(2, 1, 2)
            try:
                plot_score_density(
                    ax, df,
                    score=score,
                    title=score,
                    xlabel=score,
                    legend=dict(loc=2),
                    exclude_outlier=False
                )
            except:
                pass

            if title is not None:
                fig.suptitle(title)
            pdf.savefig(fig)
            plt.close(fig)