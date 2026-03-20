# metodos_estadisticos.py
import numpy as np
from scipy import stats
from config import PERMUTATIONS, ALPHA


def test_t_student(x, y):
    _, p_val = stats.ttest_ind(x, y, equal_var=False)
    return p_val < ALPHA


def test_wilcoxon(x, y):
    _, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')
    return p_val < ALPHA


def test_hodges_lehmann(x, y):
    m, n = len(x), len(y)
    mn = m * n

    mean_u = mn / 2
    std_u = np.sqrt(mn * (m + n + 1) / 12)
    z_alpha = stats.norm.ppf(1 - ALPHA / 2)

    lower_idx = max(0,      int(np.round(mean_u - z_alpha * std_u)) - 1)
    upper_idx = min(mn - 1, int(np.round(mean_u + z_alpha * std_u)) - 1)

    diffs = (y[:, None] - x).flatten()

    # partition garantiza que el elemento en lower_idx es el correcto
    # y todo lo que está a su izquierda es <= que él (sin ordenar ese lado)
    partitioned = np.partition(diffs, [lower_idx, upper_idx])

    ci_lower = partitioned[lower_idx]
    ci_upper = partitioned[upper_idx]

    return not (ci_lower <= 0 <= ci_upper)


def test_permutacion(x, y, estadistico='media'):
    def diff_medias(a, b, axis=0):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    def diff_medianas(a, b, axis=0):
        return np.median(a, axis=axis) - np.median(b, axis=axis)

    def diff_trim(a, b, axis=0):
        return stats.trim_mean(a, 0.1, axis=axis) - stats.trim_mean(b, 0.1, axis=axis)

    if estadistico == 'media':
        stat_func = diff_medias
    elif estadistico == 'mediana':
        stat_func = diff_medianas
    else:
        stat_func = diff_trim

    res = stats.permutation_test((x, y), stat_func, permutation_type='independent',
                                 n_resamples=PERMUTATIONS, alternative='two-sided', vectorized=True)
    return res.pvalue < ALPHA