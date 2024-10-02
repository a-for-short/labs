import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, tstd
from scipy.interpolate import interpolate
from scipy.signal import find_peaks
import seaborn as sns
import mplcursors

df = pd.read_csv('all-calibration-rel-to-water.csv',
                 delimiter=';', decimal=',',
                 engine='python', encoding="latin-1",
                 skiprows=[0,1], nrows=250)
df_ex = pd.read_csv('ex2.csv',
                           delimiter=';', decimal=',',
                           engine='python', encoding="latin-1",
                           skiprows=[0,1], nrows=150)
del df['Unnamed: 14']
del df_ex['Unnamed: 4']

col_names = []
exer_names = []
for i in range(1, 8):
    col_names.append( f'{i}-wl' )
    col_names.append( f'{i}-abs' )
for i in range(1, 3):
    exer_names.append( f'{i}-wl-ex' )
    exer_names.append( f'{i}-abs-ex' )

df.columns = col_names
df_ex.columns = exer_names

added_labels = set()

# Plot calibration series
for i in range(len(col_names) // 2):
    label = 'Calibration series'
    if label not in added_labels:
        plt.plot(df[names[i * 2]], df[names[i * 2 + 1]], color='gray', label=label)
        added_labels.add(label)
    else:
        plt.plot(df[names[i * 2]], df[names[i * 2 + 1]], color='gray')

# Plot exercises
for i in range(len(exer_names) // 2):
    label = 'Exercises'
    if label not in added_labels:
        plt.plot(df_ex[exer_names[i * 2]], df_ex[exer_names[i * 2 + 1]], color='red', label=label)
        added_labels.add(label)
    else:
        plt.plot(df_ex[exer_names[i * 2]], df_ex[exer_names[i * 2 + 1]], color='red')

plt.legend()
mplcursors.cursor()
plt.xlim(350, 500)
plt.show()