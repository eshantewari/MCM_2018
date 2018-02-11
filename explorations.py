import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


statepath = gd.get_path('NM')

azdata = pd.read_csv(statepath, header=0, index_col=0)
datadic = gd.create_data_dictionary('ProblemCDataDic.csv')
print(datadic.columns)

factors = ['FFTCB', 'TETXB', 'NUETB', 'RETCB', 'ELNIB', 'ELISB']

plt.figure(figsize=(14,8))

fig, ax = plt.subplots()

fig.set_size_inches(14,8)


for factor in factors:
    ax.plot(azdata[factor], label=datadic.loc[factor, 'Description'])

ax.plot(azdata['CLTCB'] + azdata['NNTCB'] + azdata['PMTCB'], label='sum')
ax.plot(azdata['TETXB']-azdata['ELISB'], label='Real total')

ax.legend()


plt.show()