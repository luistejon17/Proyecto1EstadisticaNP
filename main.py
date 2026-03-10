# main.py
import pandas as pd
import time
from config import SIMULATIONS
from distribuciones import generar_muestras
from metodos_estadisticos import (test_t_student, test_wilcoxon,
                                  test_hodges_lehmann, test_permutacion)


def ejecutar_simulaciones():
    resultados = []

    for escenario in range(1, 2):
        print(f"Ejecutando escenario {escenario}...")


        rechazos = {
            't_student': 0,
            'wilcoxon': 0,
            'hodges_lehmann': 0,
            'perm_media': 0,
            'perm_mediana': 0,
            'perm_trim': 0
        }

        tiempos = {k: 0.0 for k in rechazos.keys()}

        for _ in range(SIMULATIONS):
            x, y = generar_muestras(escenario)

            # T-Student
            t0 = time.time()
            if test_t_student(x, y): rechazos['t_student'] += 1
            tiempos['t_student'] += time.time() - t0

            # Wilcoxon
            t0 = time.time()
            if test_wilcoxon(x, y): rechazos['wilcoxon'] += 1
            tiempos['wilcoxon'] += time.time() - t0

            # Hodges-Lehmann
            t0 = time.time()
            if test_hodges_lehmann(x, y): rechazos['hodges_lehmann'] += 1
            tiempos['hodges_lehmann'] += time.time() - t0

            # Permutación Media
            t0 = time.time()
            if test_permutacion(x, y, 'media'): rechazos['perm_media'] += 1
            tiempos['perm_media'] += time.time() - t0

            # Permutación Mediana
            t0 = time.time()
            if test_permutacion(x, y, 'mediana'): rechazos['perm_mediana'] += 1
            tiempos['perm_mediana'] += time.time() - t0

            # Permutación Media Truncada
            t0 = time.time()
            if test_permutacion(x, y, 'trim'): rechazos['perm_trim'] += 1
            tiempos['perm_trim'] += time.time() - t0

        potencia = {k: v / SIMULATIONS for k, v in rechazos.items()}

        fila = {'escenario': escenario}
        fila.update({f'potencia_{k}': v for k, v in potencia.items()})
        fila.update({f'tiempo_{k}': v for k, v in tiempos.items()})

        resultados.append(fila)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('resultados_potencia.csv', index=False)
    print("Simulaciones completadas. Resultados guardados en 'resultados_potencia.csv'")


if __name__ == '__main__':
    ejecutar_simulaciones()