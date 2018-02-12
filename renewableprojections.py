import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import graphinitialdata as gd

def renewable_project(renewable):

    # Models, graphs, returns renewable cost projections by state
    # get data
    data = pd.Series(np.random.standard_exponential((50)), index = list(np.arange(1960, 2010, 1)))

    # Take log
    logdata = np.log(data.values)
    print(logdata)
    x = np.array(data.index)

    # Fit x, logy, create model data
    fit = np.polyfit(x, logdata, 1, w=np.sqrt(data.values))
    years = np.arange(1960, 2051, 1)
    modeldata = math.e ** (fit[0] * years + fit[1])

    if __name__ == '__main__':
        fig, ax = plt.subplots()
        ax.plot(data, color='blue', label='Observed {} Cost Data'.format(renewable), alpha=0.8)
        ax.plot(years, modeldata, color='red', label='Projected {} Cost Data'.format(renewable), linestyle='dotted', alpha=0.8)
        ax.set(title='Projections for {} Data'.format(renewable))
        ax.set(xlabel='Year', ylabel='Cents per KWH')
        ax.legend()
        plt.show()

renewable_project('Nuclear')