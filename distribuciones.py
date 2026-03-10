# distribuciones.py
import numpy as np
from scipy import stats
from config import M_SIZE, N_SIZE


def generar_muestras(escenario):
    if escenario == 1:
        x = stats.norm.rvs(loc=0, scale=1, size=M_SIZE)
        y = stats.norm.rvs(loc=0, scale=1, size=N_SIZE)
    elif escenario == 2:
        x = stats.pareto.rvs(b=2.1, scale=0.8, size=M_SIZE)
        y = stats.pareto.rvs(b=2.1, scale=0.8, size=N_SIZE)
    elif escenario == 3:
        x = stats.gamma.rvs(a=1.4, scale=0.8, size=M_SIZE)
        y = stats.gamma.rvs(a=1.4, scale=0.8, size=N_SIZE)
    elif escenario == 4:
        x = stats.norm.rvs(loc=0, scale=1, size=M_SIZE)
        y = stats.norm.rvs(loc=0.33, scale=1, size=N_SIZE)
    elif escenario == 5:
        x = stats.norm.rvs(loc=0, scale=1, size=M_SIZE)
        y = stats.norm.rvs(loc=0.67, scale=1, size=N_SIZE)
    elif escenario == 6:
        x = stats.norm.rvs(loc=0, scale=1, size=M_SIZE)
        y = stats.norm.rvs(loc=1, scale=1, size=N_SIZE)
    elif escenario == 7:
        x = stats.norm.rvs(loc=0, scale=1, size=M_SIZE)
        y = stats.norm.rvs(loc=1, scale=1, size=N_SIZE)
        outlier_val = 1 + 10 * 1
        y[np.random.choice(N_SIZE, 4, replace=False)] = outlier_val
    elif escenario == 8:
        x = stats.gamma.rvs(a=1.4, scale=0.8, size=M_SIZE)
        y = stats.pareto.rvs(b=2.1, scale=0.8, size=N_SIZE)
    elif escenario == 9:
        x = stats.gamma.rvs(a=1.1, scale=1.6, size=M_SIZE)
        y = stats.pareto.rvs(b=2.02, scale=1, size=N_SIZE)
    elif escenario == 10:
        x = stats.gamma.rvs(a=0.44, scale=4.5, size=M_SIZE)
        y = stats.pareto.rvs(b=2.005, scale=1, size=N_SIZE)
    elif escenario == 11:
        x = stats.gamma.rvs(a=1, scale=1.94, size=M_SIZE)
        y = stats.pareto.rvs(b=2.001, scale=1, size=N_SIZE)
    elif escenario == 12:
        x = stats.gamma.rvs(a=1.6, scale=2.5, size=M_SIZE)
        y = stats.norm.rvs(loc=4.2, scale=2.8, size=N_SIZE)
    elif escenario == 13:
        x = stats.gamma.rvs(a=3, scale=4, size=M_SIZE)
        y = stats.norm.rvs(loc=12.5, scale=5.5, size=N_SIZE)
    else:
        raise ValueError("Escenario no válido")

    return x, y