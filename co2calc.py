import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd

## This section will calculate CO2 totals

# Calculates equivilant CO2 emissions based on some method of aggregating the data,
# by default total end use (this analysis excludes exports)
def calc_CO2(state, agg = 'TCB'):
    statedata, datadic = gd.get_data(state)
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

    # Define result dataframe
    all_co2 = pd.DataFrame(np.zeros((50, len(statelist))),
                           index=np.arange(1960, 2010, 1),
                           columns=statelist)
    # Fill dataframe
    for state in statelist:
        all_co2[state] = calc_CO2(state, agg)

    return all_co2

# Graph totals
def graph_co2_totals(co2data, pop_adj = True):
    fig, ax = plt.subplots()
    for state in co2data.columns:
        if pop_adj:
            ax.plot(10**6 * co2data[state].divide(gd.get_pop_data(state)),
                    label=state + ' Emissions')
        else:
            ax.plot(co2data[state], label=state + ' Emissions')


    if pop_adj:
        ax.set(title='Per Capita Greenhouse Gas Emissions Over Time',
               xlabel='Year', ylabel='CO2 Equivilant Per Capita (Tons)')
    else:
        ax.set(title='Greenhouse Gas Emissions Over TIme', xlabel = 'Year', ylabel = 'CO2 Equivilant (Megatons)')
    ax.legend()
    plt.show()


## This section will calculate CO2 emissions by sector
def calc_sector_emissions(state, agg = 'TCB'):
    statedata, datadic = gd.get_data(state)
    co2factors = pd.read_csv('CO2Factors.csv', header=0, index_col=0)

    # Create column list of factors
    factors = []
    for tech in co2factors.index:
        if tech != 'Nuclear Energy':
            factors.append(co2factors.loc[tech, 'Code'] + agg)
        else:
            factors.append(co2factors.loc[tech, 'Code'] + 'EGB')

    # Create dataframe of outputs and add data
    co2count = pd.DataFrame(np.zeros((len(statedata.index), len(factors))), index=statedata.index, columns=factors)
    for tech, factor in zip(co2factors.index, factors):
        co2count[factor] = co2factors.loc[tech, 'kgCO2eq/bilBTU'] * statedata[factor]

    return co2count/10**9

def calc_sector_mortality(statedata, agg = 'TCB'):

    dfactors = pd.read_csv('MortalityFactors.csv', header=0, index_col=0)

    # Create column list of factors
    factors = []
    for tech in dfactors.index:
        if tech != 'Nuclear Energy':
            factors.append(dfactors.loc[tech, 'Code'] + agg)
        else:
            factors.append(dfactors.loc[tech, 'Code'] + 'EGB')

    # Create dataframe of outputs and add data
    dcount = pd.DataFrame(np.zeros((len(statedata.index), len(factors))), index=statedata.index, columns=factors)
    for tech, factor in zip(dfactors.index, factors):
        dcount[factor] = dfactors.loc[tech, 'Deaths/Bil BTU'] * statedata[factor]

    return np.floor(dcount)


if __name__ == '__main__':

    mort_sector = calc_sector_mortality(gd.get_data('TX')[0])
    mort_total = mort_sector.sum(axis=1)
    print(mort_total)
    plt.plot(mort_total.divide(gd.get_pop_data('TX')))
    plt.show()

    co2 = calc_all_CO2(['AZ', 'TX', 'CA', 'NM'])
    graph_co2_totals(co2)
    graph_co2_totals(co2, pop_adj = False)