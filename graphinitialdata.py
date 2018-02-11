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

# Get data and data dictionary
def get_data(statecode):
    statedata = pd.read_csv(get_path(statecode), header=0, index_col=0)
    datadic = create_data_dictionary('ProblemCDataDic.csv')
    return statedata, datadic

# Get only population data from the state
def get_pop_data(statecode):
    statedata, datadic = get_data(statecode)
    return statedata['TPOPP']

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
def graph_multiple_attributes(attributes, statedata, datadic):

    fig, ax = plt.subplots()

    fig.set_size_inches(14, 8)

    # Get colors and state data
    colors = plt.cm.rainbow(np.linspace(0, 1, len(attributes)))

    # Run through attributes list
    for color, attribute in zip(colors, attributes):

        # Ignore if attribute isn't one of the features, this will happen occasionally
        if attribute not in statedata.columns:
            continue

        ax.plot(statedata.index, statedata.loc[:, attribute], color=color,
                label= datadic.loc[attribute, 'Description'] + datadic.loc[attribute, 'Unit'])

    ax.set(title = 'Attributes in State Over Time', ylabel = 'Use', xlabel='Year')
    ax.legend()
    plt.show()

# Run examples of graphing functions and create the data dictionary
if __name__ == '__main__':
    statedata, datadic = get_data('TX')
    print(datadic.columns)
    graph_attribute('BMTCB', ['AZ', 'CA', 'NM', 'TX'])
    attributes = [index for index in datadic.index if 'ISB' in index]
    print(attributes)
    graph_multiple_attributes(attributes, statedata, datadic)