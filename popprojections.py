import math
import numpy as np
import pandas as pd
import pylab
import matplotlib.pyplot as plt
import graphinitialdata as gd
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr

# Define sigmoid function
def sigmoid(x, x0, k, a, c):
    y = a / (1 + np.exp(-k * (x - x0))) + c
    return y

# Models, graphs, returns population projections by state
def create_population_projection(state):

    # get popdata
    popdata = gd.get_pop_data(state)

    # Take log
    logpopdata = np.log(popdata.values)
    x = np.array(popdata.index)

    # Fit x, logy, create model data
    fit = np.polyfit(x, logpopdata, 1, w=np.sqrt(popdata.values))
    years = np.arange(1960, 2051, 1)
    modeldata = math.e**(fit[0]*years + fit[1])

    # Plot if it's the main thing
    if __name__ == '__main__':
        fig, ax = plt.subplots()
        ax.plot(popdata, color='blue', label='Observed Population Data', alpha=0.8)
        ax.plot(years, modeldata, color='red', label='Projected Population Data', linestyle='dotted', alpha=0.8)
        ax.set(title='Population Projections for {}'.format(state))
        ax.set(xlabel='Year', ylabel='Population (Thousands)')
        ax.legend()
        plt.show()

    # Return projected data
    modeldata = pd.Series(modeldata, index=years)
    modeldata = modeldata.ewm(alpha=0.05).mean()

    return modeldata[[year for year in modeldata.index if int(year)>2009]]

def get_all_pop_data(state):

    realdata = gd.get_pop_data(state)
    projdata = create_population_projection(state)

    return realdata.append(projdata)

def create_gdp_data(state):

    # get gd[data
    gdpdata = gd.get_gdp_data(state)

    # Take log
    x = np.array(gdpdata[gdpdata.notnull()].index)
    loggdpdata = np.log(gdpdata[x].values)

    # Fit x, logy, create model data
    fit = np.polyfit(x, loggdpdata, 1, w=np.sqrt(gdpdata[x].values))
    years = np.arange(1960, 2051, 1)
    modeldata = math.e**(fit[0]*years + fit[1])

    # Plot if it's the main thing
    if __name__ == '__main__':
        fig, ax = plt.subplots()
        ax.plot(gdpdata, color='blue', label='Observed GDP Per Cap Data', alpha=0.8)
        ax.plot(years, modeldata, color='red', label='Projected GDP Per Capita Data', linestyle='dotted', alpha=0.8)
        ax.set(title='GDP Per Capita Projections for {}'.format(state))
        ax.set(xlabel='Year', ylabel='GDP (Thousands of Dollars)')
        ax.legend()
        plt.show()

    # Return projected data
    modeldata = pd.Series(modeldata, index=years)
    modeldata = modeldata.ewm(alpha=0.05).mean()

    return modeldata[[year for year in modeldata.index if int(year)>2009]]

def get_all_gdp_data(state):

    realdata = gd.get_gdp_data(state)
    projdata = create_gdp_data(state)

    return realdata.append(projdata)