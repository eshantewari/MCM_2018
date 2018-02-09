import csv
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm

# Format data by state. Will turn it from a big list into a dataframe with the index as the year
# and the features as the columns.
def formatinitialdata(path, statecode):

    # Get state data
    data = pd.read_csv(path, header=0)
    statedata = data.loc[data.loc[:, 'StateCode'] == statecode]

    # Get new indexes
    years = list(set(data['Year']))
    features = list(set(statedata['MSN']))
    features.sort()

    # Reindex and fill new dataframe with the data
    formatteddata = pd.DataFrame(index = years, columns=[])

    for feature in tqdm(features):
        indexes = statedata.loc[:, 'MSN'] == feature
        placeholder = pd.Series(data=statedata.loc[indexes, 'Data'].values,  index = statedata.loc[indexes, 'Year'])
        formatteddata[feature] = placeholder

    return formatteddata


# This will run if you put the data into your working directory as a csv - I've uploaded the csv to git
def Run_formatting(path, statecode):
    outputpath = 'MCM' + str(statecode) + 'Data.csv'
    data = formatinitialdata(path, statecode)
    print(data)
    data.to_csv(outputpath, sep=',',  line_terminator='\n')
    return None

# Run the function for all four states
for statecode in ['TX', 'AZ', 'CA', 'NM']:
    print(statecode)
    Run_formatting('ProblemCData.csv', statecode)
    print('Finished with {}'.format(statecode))


