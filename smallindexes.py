import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
from sklearn import linear_model, decomposition
from scipy.stats.stats import pearsonr

### Define Global Variables
# These are some sectors
sectordic = {'CC': 'Comercial', 'IC': 'Industrial', 'RC':'Residential', 'AC': 'Transportation',
             'TC': 'Total Primary Consumption', 'TX': 'Total End Use', 'EG': 'Total Generation',
             'EI': 'Electric Sector Consumption'}

# These are some sources
sourcedic = {'CL':'Coal', 'NN': 'Natural Gas', 'PA': 'Petroleum Products',
             'NU': "Nuclear Electric Power", 'EM': 'Fuel Ethanol',
             'GE': 'Geothermal', 'HY': 'Hydroelectric', 'SO': 'Solar Thermal',
             'WD': 'wood', 'WS': 'Biomass waste', 'ES': 'Electricity Sales',
             'LO': 'Electrical System Losses', 'WY': 'Wind'}

# For labeling, create a unit dictionary
unitdic = {'B': 'Billions BTU', 'V': 'Millions Dollars'}

### Helper functions which adjust for population, GDP, inflation

# Adjusts for population per capita
def adjust_for_population(statedata, unit):
    factors = [factor for factor in statedata.columns if factor[-1] == unit]
    statedata[factors] = statedata[factors].divide((statedata['TPOPP']*1000), axis=0)
    return statedata[factors]

# Adjusts for GDP per capita
def adjust_for_gdp(statedata, unit):
    factors = [factor for factor in statedata.columns if factor[-1] == unit]
    statedata[factors] = statedata[factors].divide((1000000*statedata['GDPRX']), axis=0)
    return statedata

# Adjusts for inflation over time. The unit should probably be V or K if it is to make any sense.
def adjust_for_inflation(statedata, unit):
    factors = [factor for factor in statedata.columns if factor[-1] == unit]
    inflation = pd.read_csv('Inflation.csv', header=0, index_col=0)
    statedata[factors] = statedata[factors].multiply(inflation['Value in 2009 Dollars'], axis=0)
    return statedata

### Actually graphing things

# Helper function, will either graph a profile of the given sector's energy use in BTU
# or the given source's energy use in BTU. It does the bulk of the work to graph.
def graph_either_profile(attributelist, given, statedata, datadic, min, state, pop_adj, inf_adj, unit):

    # Adjust for state population
    if pop_adj == True:
        statedata = adjust_for_population(statedata, unit)
    if inf_adj == True and unit == 'V':
        statedata = adjust_for_inflation(statedata, unit)

    # Get rid of factors which are less than the min times the mean of the attributelist
    means = statedata[attributelist].mean()
    attributelist.sort(key = lambda x: means[x], reverse=True)
    for factor in means.index:
        if means[factor] < means.mean()*min:
            attributelist.remove(factor)

    # Get colors
    colors = plt.cm.jet(np.linspace(0, 1, len(attributelist)))

    # Create fig, ax
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)

    # Set labels and title, depending on some of the options

    if inf_adj and unit == 'V':
        newunit = unitdic[unit] + ', Adjusted for Inflation'
    else:
        newunit = unitdic[unit]

    # Set axis
    if pop_adj:
        ax.set(xlabel='Year', ylabel='Consumption Per Capita ({})'.format(newunit))
    else:
        ax.set(xlabel='Year', ylabel='Consumption ({})'.format(newunit))
    if given in sourcedic:
        ax.set(title = '{} {} Energy Use, 1960-2009'.format(state, sourcedic[given]))
    elif given in sectordic:
        ax.set(title = '{} {} Energy Sources, 1960-2009'.format(state, sectordic[given]))
    else:
        print('You probably inputted the wrong sector or source, which is why the title is missing')

    # Graph the main contributors on top of each other, in different colors, doing the first one manually
    ax.fill_between(np.arange(1960, 2010, 1),
                    [0]*50,
                    statedata[attributelist[0]],
                    color=colors[0],
                    label = datadic.loc[attributelist[0], 'Description'],
                    alpha = 0.4)

    # Do the rest automatically
    i = 1
    while i < len(attributelist):

        #Be careful about labelling
        try:
            label = datadic.loc[attributelist[i], 'Description']
        except:
            label = sourcedic[attributelist[i][0:2]] + sectordic[attributelist[i][2:4]]

        # Graph
        ax.fill_between(np.arange(1960, 2010, 1),
                        statedata[attributelist[0:i+1]].sum(axis=1),
                        statedata[attributelist[0:i]].sum(axis=1),
                        color = colors[i],
                        label = label,
                        alpha = 0.4)
        i += 1

    ax.legend()
    plt.show()

    return statedata

# Will graph a profile of the sector's energy use, in the given unit, and returns the relevant attributes
def graph_sector_profile(sector, statedata, datadic, min, state, pop_adj = True, inf_adj = True, unit='B'):

    # Find the right attributes/factors
    attributelist = [factor for factor in statedata.columns if factor[0:2] in sourcedic and factor[2:4] == sector
                     and factor[-1] == unit]

    graph_either_profile(attributelist, sector, statedata, datadic, min, state, pop_adj, inf_adj, unit)

    return attributelist, statedata

# Will graph a profile of the source's energy use, in the given unit, and returns the relevant attributes
def graph_source_profile(source, statedata, datadic, min, state, pop_adj = True, inf_adj = True, unit='B'):

    # Find the right attributes/factors
    attributelist = [factor for factor in statedata.columns if factor[0:2] == source and factor[2:4] in sectordic
                     and factor[-1] == unit]

    statedata = graph_either_profile(attributelist, source, statedata, datadic, min, state, pop_adj, inf_adj, unit)

    return attributelist, statedata

# Runs everything at once
def create_sector_profile(state, sector, min = 0.01, pop_adj = True, inf_adj=True, unit='B'):
    statedata, datadic = gd.get_data(state)
    attributelist, statedata = graph_sector_profile(sector, statedata, datadic, min, state, pop_adj=pop_adj, inf_adj=inf_adj, unit=unit)
    return statedata[attributelist]

def create_source_profile(state, source, min=0.01, pop_adj=True, inf_adj=True, unit='B'):
    statedata, datadic = gd.get_data(state)
    attributelist, statedata = graph_source_profile(source, statedata, datadic, min, state, pop_adj=pop_adj, inf_adj=inf_adj, unit=unit)
    return statedata[attributelist]

## Here, you can just use my template to run things
if __name__=='__main__':
    statelist = ['AZ', 'TX', 'NM', 'CA']
    print(create_source_profile('TX', 'CL', min=0.01, inf_adj=False, unit='V'))
    print(create_source_profile('TX', 'CL', min=0.01, inf_adj=True, unit='V'))



#### Ignore this if you're not asher
def check_inflation_adj(state, attrb):

    moneyattrb = attrb + 'V'
    btuattrb = attrb + 'B'

    statedata, datadic = gd.get_data(state)
    fig1, ax1 = plt.subplots()
    corr1 = ((statedata[[moneyattrb, btuattrb]] -  statedata[[moneyattrb, btuattrb]].mean())/statedata[[moneyattrb, btuattrb]].std()).cov().loc[moneyattrb, btuattrb]
    ax1.set(title='Money and BTU Not Adjusted in {}, corr = {}'.format(state, corr1))
    ax1.plot(statedata[attrb + 'V']*1000, label='Money', color='blue')
    ax1.plot(statedata[attrb + 'B'], label='BTU', color='orange')
    ax1.legend()

    statedata = adjust_for_inflation(statedata, 'V')
    fig2, ax2 = plt.subplots()
    corr2 = ((statedata[[moneyattrb, btuattrb]] -  statedata[[moneyattrb, btuattrb]].mean())/statedata[[moneyattrb, btuattrb]].std()).cov().loc[moneyattrb, btuattrb]
    ax2.set(title='Money and BTU Inflation Adjusted in {}, corr = {}'.format(state, corr2))
    ax2.plot(statedata[attrb + 'V']*1000, label='Money', color='blue')
    ax2.plot(statedata[attrb + 'B'], label='BTU', color='orange')
    ax2.legend()
    plt.show()
