import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import popprojections as pp
import smallindexes as si

### This script calculates and graphs CO2 Emissions and Mortality rates both by industry and in total.

ylabel_dic = {'T': 'Megatons of CO2 Equivilant', 'D': 'Deaths caused by Particulate Pollution',
         'V':'Millions of Dollars of Damage from CO2 Equivilant Over the next Decade'}
title_dic = {'T': 'GHG Emissions', 'D': 'Annual Deaths from Air Pollution' , 'V': 'Monetary Cost of GHG Emissions'}

## This section will calculate CO2 totals

# Calculates equivilant CO2 emissions based on some method of aggregating the data,
# by default total end use (this analysis excludes exports)
def calc_CO2(statedata, datadic, agg = 'TCB'):
    co2factors = pd.read_csv('CO2Factors.csv', header=0, index_col=0)
    co2count = pd.Series(np.zeros((len(statedata.index))), index=statedata.index)
    for tech in co2factors.index:
        factor = co2factors.loc[tech, 'Code'] + agg
        if factor in statedata.columns:
            co2count += co2factors.loc[tech, 'kgCO2eq/bilBTU']*statedata[factor]

        # Try using EGB for nuclear energy sector, which is weird
        else:
            factor = co2factors.loc[tech, 'Code'] + 'EGB'
            if factor in statedata.columns:
                co2count += co2factors.loc[tech, 'kgCO2eq/bilBTU'] * statedata[factor]
            else:
                print('Did not include {}, {} factor not in the state data file'.format(
                    tech, factor
                ))
    return co2count/10**9

# Do this for all states and label appropriately
def calc_all_CO2(statelist, agg = 'TCB'):

    years = gd.get_data(statelist[0])[0].index

    # Define result dataframe
    all_co2 = pd.DataFrame(np.zeros((len(years), len(statelist))),
                           index=years,
                           columns=statelist)
    # Fill dataframe
    for state in statelist:
        statedata, datadic = gd.get_data(state)
        all_co2[state] = calc_CO2(statedata, datadic, agg)

    return all_co2

# Graph totals
def graph_co2_totals(co2data, pop_adj = True):
    fig, ax = plt.subplots()
    for state in co2data.columns:
        if pop_adj:
            ax.plot(10**6 * co2data[state].divide(gd.get_pop_data(state).loc[co2data.index]),
                    label=state + ' Emissions')
        else:
            ax.plot(co2data[state], label=state + ' Emissions')


    if pop_adj:
        ax.set(title='Per Capita Greenhouse Gas Emissions Over Time',
               xlabel='Year', ylabel='CO2 Equivilant Per Capita (Tons)')
    else:
        ax.set(title='Greenhouse Gas Emissions Over TIme', xlabel = 'Year', ylabel = 'CO2 Equivilant (Megatons)')
    ax.legend()

## This section will calculate CO2 emissions by sector
def calc_sector_emissions(statedata, agg = 'TCB'):
    co2factors = pd.read_csv('CO2Factors.csv', header=0, index_col=0)

    # Create column list of factors
    factors = []
    for tech in co2factors.index:
        if tech == 'Nuclear Energy' and 'NU'+agg not in statedata.columns:
            factors.append(co2factors.loc[tech, 'Code'] + 'EGB')
        else:
            factors.append(co2factors.loc[tech, 'Code'] + agg)

    # Create dataframe of outputs and add data
    co2count = pd.DataFrame(np.zeros((len(statedata.index), len(factors))), index=statedata.index, columns=factors)
    for tech, factor in zip(co2factors.index, factors):
        co2count[factor] = co2factors.loc[tech, 'kgCO2eq/bilBTU'] * statedata[factor]

    return co2count/10**9

def calc_sector_mortality(statedata, agg='TCB'):

    dfactors = pd.read_csv('MortalityFactors.csv', header=0, index_col=0)

    # Create column list of factors
    factors = []
    for tech in dfactors.index:
        if tech == 'Nuclear Energy' and 'NU'+agg not in statedata.columns:
            factors.append(dfactors.loc[tech, 'Code'] + 'EGB')
        else:
            factors.append(dfactors.loc[tech, 'Code'] + agg)

    # Create dataframe of outputs and add data
    dcount = pd.DataFrame(np.zeros((len(statedata.index), len(factors))), index=statedata.index, columns=factors)
    for tech, factor in zip(dfactors.index, factors):
        dcount[factor] = dfactors.loc[tech, 'Deaths/Bil BTU'] * statedata[factor]

    return np.floor(dcount)


# Graph emisssions profile, either for mortality, co2, or cost per co2
def graph_emissions_profile(state, unit, min=0.01, pop_adj=True, inf_adj=True, sum = False, agg = 'TCB'):

    # Get statedata
    statedata, datadic = gd.get_data(state)

    # Get mortality, Co2, or co2 cost data
    if unit == 'D':
        data = calc_sector_mortality(statedata, agg)
    elif unit == 'T':
        data = calc_sector_emissions(statedata, agg)
    elif unit == 'V':
        data = calc_sector_emissions(statedata, agg)*11 # This 11 is from the EPA, it's the social cost of carbon per metric ton
    else:
        print('You entered the wrong type of unit: the only types are T, D, and V')
        return None

    return graph_emissions_profile(data, unit, min, pop_adj, inf_adj, sum, agg)

def graph_emissions_profile_v2(data, datadic, state, unit='B', min=0.01, pop_adj=True,
                               inf_adj=True, sum = False, agg = 'TCB'):

    # Adjust for state population
    if pop_adj == True:
        popdata = pp.get_all_pop_data(state)
        data = data.divide(1000*popdata.loc[data.index], axis=0)
    if inf_adj == True and unit == 'V':
        infdata = gd.get_inf_data()
        data = data.multiply(infdata['Value in 2009 Dollars'], axis=0)


    # Get colors
    colors = plt.cm.jet(np.linspace(0, 1, len(data.columns)))

    # Create fig, ax
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 8)

    # Set axis labels and title, depending on some of the options
    if pop_adj:
        ax.set(xlabel='Year', ylabel=ylabel_dic[unit]+ ' Per Capita')
    else:
        ax.set(xlabel='Year', ylabel=ylabel_dic[unit])
    if sum == True:
        ax.set(title = title_dic[unit] + ', Aggregate')
    else:
        ax.set(title = title_dic[unit] + ' by Energy Source')


    # Sort factors based on their mean costs, get rid of factors which are less than the min times the mean of attributelist
    attributelist = [item for item in list(data.columns) if agg in item]
    means = data[attributelist].mean()
    attributelist.sort(key = lambda x: means[x], reverse=True)
    for factor in means.index:
        if means[factor] < means.mean()*min:
            attributelist.remove(factor)

    print(data.index)

    # Graph the main contributors on top of each other, in different colors, doing the first one manually
    ax.fill_between(data.index,
                    [0]*len(data),
                    data[attributelist[0]],
                    color=colors[0],
                    label = datadic.loc[attributelist[0], 'Description'],
                    alpha = 0.4)

    # Do the rest automatically
    i = 1
    while i < len(attributelist):

        #Be careful about labelling
        if attributelist[i] in datadic.index:
            label = datadic.loc[attributelist[i], 'Description']
        else:
            try:
                label = si.sourcedic[attributelist[i][0:2]] + si.sectordic[attributelist[i][2:4]]
            except:
                label = 'Unknown attribute {}'.format(attributelist[i])

        # Graph
        ax.fill_between(list(data.index),
                        data[attributelist[0:i+1]].sum(axis=1),
                        data[attributelist[0:i]].sum(axis=1),
                        color = colors[i],
                        label = label,
                        alpha = 0.4)
        i += 1

    ax.legend()
    plt.show()

    return data

if __name__ == '__main__':
    graph_emissions_profile_v2('TX', 'V', pop_adj=False, inf_adj=False)
