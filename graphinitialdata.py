import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Gets path of data for state
def get_path(statecode):
    return 'MCM' + str(statecode) + 'Data.csv'

# Create data dictionary, the path should be to a csv of the data dictionary from the competition
def create_data_dictionary(path):
    datadic = pd.read_csv(path, header=0, index_col=0)
    return datadic

# Graph a specific attribute of a list of states over time
def graph_attribute(attribute, statecodes):

    fig, ax = plt.subplots()

    # Get colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(statecodes)))

    # Run through statecodes list
    for color, state in zip(colors, statecodes):

        statedata = pd.read_csv(get_path(state), index_col = 0, header=0)
        ax.plot(statedata.index, statedata.loc[:, attribute], color=color, label=state)

    ax.set(title = '{} Over Time'.format(attribute), ylabel = '{} Use'.format(attribute), xlabel='Year')
    ax.legend()
    plt.show()


# Graph multiple attributes of a single state, get correlation between them, presumably they have the same units
def graph_multiple_attributes(attributes, state):

    fig, ax = plt.subplots()

    # Get colors and state data
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attributes)))
    statedata = pd.read_csv(get_path(state), index_col=0, header=0)

    # Run through attributes list
    for color, attribute in zip(colors, attributes):

        ax.plot(statedata.index, statedata.loc[:, attribute], color=color, label=attribute)

    ax.set(title = 'Attributes in {} Over Time'.format(state), ylabel = 'Use', xlabel='Year')
    ax.legend()
    plt.show()

# Run examples of graphing functions and create the data dictionary

datadic = create_data_dictionary('ProblemCDataDic.csv')

print(datadic.columns)
graph_attribute('BMTCB', ['AZ', 'CA', 'NM', 'TX'])
attributes = [index for index in datadic.index if 'TCV' in index]
graph_multiple_attributes(attributes, 'TX')