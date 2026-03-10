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
    diffs = (y[:, None] - x).flatten()
    diffs.sort()

    m, n = len(x), len(y)
    mean_u = m * n / 2
    std_u = np.sqrt(m * n * (m + n + 1) / 12)
    z_alpha = stats.norm.ppf(1 - ALPHA / 2)

    lower_idx = int(np.round(mean_u - z_alpha * std_u))
    upper_idx = int(np.round(mean_u + z_alpha * std_u))

    lower_idx = max(0, lower_idx - 1)
    upper_idx = min(len(diffs) - 1, upper_idx - 1)

    ci_lower = diffs[lower_idx]
    ci_upper = diffs[upper_idx]

    return not (ci_lower <= 0 <= ci_upper)


def test_permutacion(x, y, estadistico='media'):
    def diff_medias(a, b):
        return np.mean(a) - np.mean(b)

    def diff_medianas(a, b):
        return np.median(a) - np.median(b)

    def diff_trim(a, b):
        return stats.trim_mean(a, 0.1) - stats.trim_mean(b, 0.1)

    if estadistico == 'media':
        stat_func = diff_medias
    elif estadistico == 'mediana':
        stat_func = diff_medianas
    else:
        stat_func = diff_trim

    res = stats.permutation_test((x, y), stat_func, permutation_type='independent',
                                 n_resamples=PERMUTATIONS, alternative='two-sided')
    return res.pvalue < ALPHA