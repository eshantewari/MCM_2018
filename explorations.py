import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sys import exit

azdata, datadic = gd.get_data('AZ')
factorlist = [factor for factor in azdata.columns if factor[-1] == 'B']
datadic.loc[factorlist].to_csv('BTUDataDic.csv')


exit()


print(azdata)
azdata = pd.DataFrame(azdata.ewm().mean(), index=azdata.index, columns=azdata.columns)
print(azdata)

datadic = gd.create_data_dictionary('ProblemCDataDic.csv')
print(datadic.columns)

factors = ['FFTCB', 'TETXB', 'NUETB', 'RETCB', 'ELNIB', 'ELISB']

fig, ax = plt.subplots()

fig.set_size_inches(14,8)

for factor in factors:
    ax.plot(azdata[factor], label=datadic.loc[factor, 'Description'])

ax.plot(azdata['CLTCB'] + azdata['NNTCB'] + azdata['PMTCB'], label='sum')
ax.plot(azdata['TETXB']-azdata['ELISB'], label='Real total')

ax.legend()


plt.show()