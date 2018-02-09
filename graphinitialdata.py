import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gets path of data for state
def get_path(statecode):
    return 'MCM' + str(statecode) + 'Data.csv'

# Graph a specific attribute of a list of states over time
def graph_attribute(attribute, statecodes):

    fig, ax = plt.subplots()

    colors = plt.cm.rainbow(np.linspace(0, 1, len(statecodes)))

    # Run through statecodes list
    for color, state in zip(colors, statecodes):

        # Get colors

        statedata = pd.read_csv(get_path(state), index_col = 0, header=0)
        ax.plot(statedata.index, statedata.loc[:, attribute], color=color, label=state)

    ax.set(title = '{} Over Time'.format(attribute), ylabel = '{} Use'.format(attribute), xlabel='Year')
    ax.legend()
    plt.show()

graph_attribute('CLOCB', ['TX', 'AZ', 'CA', 'NM'])