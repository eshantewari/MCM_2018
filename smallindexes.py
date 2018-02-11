import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd

sectordic = {'CC': 'Comercial', 'IC': 'Industrial', 'RC':'Residential', 'AC': 'Transportation',
             'TC': 'Total Primary Consumption', 'TX': 'Total End Use', 'EG': 'Total Generation',
             'EI': 'Electric Sector Consumption'}

sourcedic = {'CL':'Coal', 'NN': 'Natural Gas', 'PA': 'Petroleum Products',
             'NU': "Nuclear Electric Power", 'EM': 'Fuel Ethanol',
             'GE': 'Geothermal', 'HY': 'Hydroelectric', 'SO': 'Solar Thermal',
             'WD': 'wood', 'WS': 'Biomass waste', 'ES': 'Electricity Sales',
             'LO': 'Electrical System Losses', 'WY': 'Wind'}

# adjusts for population per capita
def adjust_for_population(statedata, unit):
    factors = [factor for factor in statedata.columns if factor[-1] == unit]
    statedata[factors] = statedata[factors].divide((statedata['TPOPP']*1000), axis=0)
    return statedata[factors]

statedata, datadic = gd.get_data('TX')
print(statedata['TPOPP'], statedata['TETXB'])
statedata = adjust_for_population(statedata, 'B')
print(statedata['TETXB'])

# Helper function, will either graph a profile of the given sector's energy use in BTU
# or the given source's energy use in BTU. It does the bulk of the work to graph.
def graph_either_profile(attributelist, given, statedata, datadic, min, state, pop_adj=True):

    # Adjust for state population
    if pop_adj == True:
        statedata = adjust_for_population(statedata, 'B')*1000

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
    if pop_adj:
        ax.set(xlabel='Year', ylabel='Consumption Per Capita (Millions BTU)')
    else:
        ax.set(xlabel='Year', ylabel='Consumption (Billions BTU)')
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
        ax.fill_between(np.arange(1960, 2010, 1),
                        statedata[attributelist[0:i+1]].sum(axis=1),
                        statedata[attributelist[0:i]].sum(axis=1),
                        color = colors[i],
                        label = datadic.loc[attributelist[i], 'Description'],
                        alpha = 0.4)
        i += 1

    ax.legend()
    plt.show()


# Will graph a profile of the sector's energy use, in BTU
def graph_sector_profile(sector, statedata, datadic, min, state='', pop_adj = True):

    # Find the right attributes/factors
    attributelist = [factor for factor in statedata.columns if factor[0:2] in sourcedic and factor[2:4] == sector
                     and factor[-1] == 'B']

    graph_either_profile(attributelist, sector, statedata, datadic, min, state, pop_adj)



def graph_source_profile(source, statedata, datadic, min, state='', pop_adj = True):

    # Find the right attributes/factors
    attributelist = [factor for factor in statedata.columns if factor[0:2] == source and factor[2:4] in sectordic
                     and factor[-1] == 'B']

    graph_either_profile(attributelist, source, statedata, datadic, min, state, pop_adj)



def create_sector_profile(state, sector, pop_adj = True):
    statedata, datadic = gd.get_data(state)
    graph_sector_profile(sector, statedata, datadic, 0.01, state, pop_adj = True)

def create_source_profile(state, source):
    statedata, datadic = gd.get_data(state)
    graph_source_profile(source, statedata, datadic, 0.01, state, pop_adj = True)


create_sector_profile('AZ', 'CC')
create_source_profile('AZ', 'NN')