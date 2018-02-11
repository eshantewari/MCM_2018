import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import smallindexes as si
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

# Computes the average squared correlation between an explanatory variable and a sector's consumption in BTU of
# different fuels
def check_squared_corr(covm, sector, explan, unit = 'B'):
    factors = [factor for factor in covm.index if factor[2:4] == sector and factor[-1] == unit]
    cors = covm.loc[factors, explan]
    cors = cors**2
    return cors.mean()

# Checks which variables of the explans has the highest correlation for a given state and unit
def check_all_squared_corrs(statecode, explans, unit='B'):

    # Get data and calculate covariance matrix
    df = pd.read_csv(gd.get_path(statecode), header=0, index_col=0)
    df_norm = (df - df.mean()) / df.std()
    covm = df_norm.cov()

    # Initialize dataframe which will be returned
    results = pd.DataFrame(index=si.sectordic, columns=explans)

    # Do analysis for all the sectors
    for sector in si.sectordic:
        for explan in explans:
            results.at[sector, explan] = check_squared_corr(covm, sector, explan, unit = 'B')

    return results

# Run for all four states - upon visual inspection, it becomes clear that population is significantly more correlated with any outcomes we're interested in than GDP is.
for statecode in ['AZ', 'CA', 'NM', 'TX']:
    print(check_all_squared_corrs(statecode, ['GDPRX', 'TPOPP']))




# Performs PCA, transforms data to n_dim dimensional object, reports what it's doing
def perform_PCA(statecode, n_dim, unitcode):

    # Get data
    statedata = pd.read_csv(gd.get_path(statecode), header=0, index_col=0)

    statedata = statedata.loc[:, [MSN for MSN in statedata.columns if unitcode == MSN[-1]]]

    statedata = statedata.fillna(0)

    # PCA object
    pca = PCA(n_components = n_dim)

    # Create new columns, which tell you which PCA dim it is
    newcolumns = []
    i = 1
    while i <= n_dim:
        newcolumns.append(str('PCA-'+str(i)))
        i += 1

    # Fit PCA on statedata
    pca.fit(normalize(statedata, axis=0))
    transformed_data = pca.transform(normalize(statedata, axis=0))
    transformed_data = pd.DataFrame(transformed_data, index=statedata.index, columns=newcolumns)

    # Get back what the PCA did
    i = np.identity(statedata.shape[1])
    print(i.shape)
    coefs = pca.transform(i)

    coefs = pd.DataFrame(data = coefs, columns=newcolumns, index=statedata.columns)

    return transformed_data, pca, coefs

newdata, pcaobject, coefs = perform_PCA('TX', 2, 'B')
normcoefs = pd.DataFrame(normalize(coefs), columns=coefs.columns, index=coefs.index)
datadic = gd.create_data_dictionary('ProblemCDataDic.csv')
print(datadic.loc[normcoefs.loc[abs(normcoefs['PCA-1']) > 0.83, ['PCA-1']].index])
