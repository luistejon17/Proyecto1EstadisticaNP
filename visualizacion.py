# visualizacion.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def graficar_potencias():
    df = pd.read_csv('resultados_potencia.csv')

    cols_potencia = [col for col in df.columns if 'potencia' in col]

    df_melted = df.melt(id_vars=['escenario'], value_vars=cols_potencia,
                        var_name='Metodo', value_name='Potencia')

    df_melted['Metodo'] = df_melted['Metodo'].str.replace('potencia_', '')

    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x='escenario', y='Potencia', hue='Metodo')
    plt.title('Potencia Estimada por Método y Escenario')
    plt.ylabel('Potencia')
    plt.xlabel('Escenario')
    plt.axhline(0.05, color='red', linestyle='--', label='Nivel de significancia (0.05)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('grafico_potencias.png')
    plt.show()


def graficar_distribuciones(x, y, titulo):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(x, color='blue', alpha=0.5, label='Muestra X', kde=True)
    sns.histplot(y, color='orange', alpha=0.5, label='Muestra Y', kde=True)
    plt.title(f'Histogramas - {titulo}')
    plt.legend()

    plt.subplot(1, 2, 2)
    datos = pd.DataFrame({'Valor': np.concatenate([x, y]),
                          'Muestra': ['X'] * len(x) + ['Y'] * len(y)})
    sns.boxplot(x='Muestra', y='Valor', data=datos)
    plt.title(f'Boxplots - {titulo}')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    graficar_potencias()