import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import co2calc as co2
import popprojections as pop
import climateprojections as climate
import renewableprojections as renew
from sys import exit
from sklearn.linear_model import *
from sklearn.svm import SVR


## Helper functions
# exponentially weight in reverse training data
def weight_training_data(data, alpha):
    # reverse
    data = data.reindex(index=data.index[::-1])
    #weight
    data = data.ewm(alpha=alpha).mean()
    #reverse again
    data = data.reindex(index=data.index[::-1])
    return data

    # Fill NANs with earliest observed value
def filler(data):
    for factor in data.columns:
        bool = data.loc[:, factor].notnull()
        earliest = min(data.index[bool])
        bool = [not i for i in bool]
        data.at[bool, factor] = data.loc[earliest, factor]
    return data

## Steal other stuff
axisdic = {'B': 'BTU (Billions)', 'D':'Annual Pollution Deaths', 'T': 'Megatons of CO2 Equivalent',
           'V':'Cost of Emissions (Millions of Dollars)'}

# Graphs predictions, whether they're CO2 Emissions, Cost, Mortality, etc
def graph_predictions(observeddata, modeldata, attributelist, datadic, path, unit='B', pop_adj = False, inf_adj = False, title='', state='',
                      earliest=1960, latest=2051):

    # Create figure, ax, colors
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    colors = plt.cm.jet(np.linspace(0, 1, len(attributelist)))

    # Adjust for earliest, latest
    observeddata = observeddata.loc[(observeddata.index > earliest) & (observeddata.index < latest)]
    modeldata = modeldata.loc[(modeldata.index > earliest) & (modeldata.index < latest)]


    # Graph all the stuff by attribute
    for color, attribute in zip(colors, attributelist):
        ax.scatter(observeddata.index, observeddata[attribute], color = color, s=0.3, alpha=1)
        ax.plot(modeldata.index, modeldata[attribute], label=final_source_dic[attribute[0:2]]+final_sector_dic[attribute[2:4]]+' Predicted', color = color, alpha=0.5)

    # Set title, axes
    ax.set(title='{} in {}, Observed and Predicted, {} to {}'.format(title, state,
                                                                     earliest,
                                                                     latest),
           xlabel = 'Years',
           ylabel = axisdic[unit])

    ax.legend()
    plt.savefig(path, bbox_inches='tight')

## Model Total End Use Growth Growth
statedata, datadic= gd.get_data('AZ')

# Define factors
factors0 = ['Population', 'Q20', 'Q40', 'Q60', 'Q80', 'GDP Per Cap']
factors1 = {'TETXB': 'Total End Use'}
factors2 = ['Solar', 'Wind', 'Coal', 'Natural Gas',
                  'Petroleum Products',	'Nuclear Electric Power', 'Oil Price']
source_dic_3 = {'NU': "Nuclear Electric Power", 'EM': 'Fuel Ethanol',
             'GE': 'Geothermal', 'HY': 'Hydroelectric', 'SO': 'Solar Thermal',
              'WY': 'Wind'}
sector_dic_3 = {'EG': 'Energy Sector Generation', 'HC':'Residential and Commercial'}

factors3 = [factor for factor in statedata.columns if factor[0:2] in source_dic_3
                  and factor[2:4] in sector_dic_3 and factor[-1] == 'B']

source_dic_4 = {'CL':'Coal', 'NG': 'Natural Gas', 'PA': 'Petroleum Products',}
sector_dic_4 = {'TX': 'Total End Use'}
factors4 = [factor for factor in statedata.columns if factor[0:2] in source_dic_4
            and factor[2:4] in sector_dic_4 and factor[-1] =='B']

final_source_dic = {**source_dic_3, **source_dic_4}
final_sector_dic = {**sector_dic_3, **sector_dic_4}
final_source_dic['TE'] = 'Total Energy Consumption'

# inputs1
def create_inputs1(state):
    inputs1 = pd.DataFrame(index=np.arange(1960, 2051, 1), columns=factors0)
    inputs1['Population'] = pop.get_all_pop_data(state)
    inputs1['GDP Per Cap'] = pop.get_all_gdp_data(state)
    inputs1[['Q20', 'Q40', 'Q60', 'Q80']] = climate.predict_climate_data(state)

    inputs1 = filler(inputs1)

    return inputs1

#outputs1
def create_outputs1(state, factors1):
    statedata, datadic = gd.get_data(state)
    return statedata[factors1]

def create_prediction1(state, factors1):
    inputs1 = create_inputs1('AZ')
    outputs1 = create_outputs1('AZ', [factor for factor in factors1])

    model1 = LinearRegression()

    model1.fit(inputs1[inputs1.index < 2005].values, outputs1.loc[outputs1.index < 2005].values)

    prediction1 = pd.DataFrame(model1.predict(inputs1), index=np.arange(1960, 2051, 1), columns=factors1)
    weights = pd.DataFrame(model1.predict(np.identity(len(inputs1.columns))), index=inputs1.columns, columns=factors1)


    graph_predictions(outputs1, prediction1, [factors for factors in factors1], datadic, 'PredictiveGraphs/'+state+'Pred1.png', title='Total Energy Consumption', state=state)

    return prediction1, weights

#inputs2
def create_inputs_2(prediction1, state):
    inputs2 = prediction1.copy(deep=True)
    inputs2['Solar'] = renew.renewable_project('Solar')
    inputs2['Solar'] = inputs2['Solar']**(-1)
    inputs2['Wind'] = renew.renewable_project('Wind')
    inputs2['Wind'] = inputs2['Wind']**(-1)
    inputs2[[ 'Coal', 'Natural Gas', 'Petroleum Products', 'Nuclear Electric Power', 'Oil Price']] = pd.read_csv(state +'_NonRen_Prices.csv', index_col=0, header=0)
    inputs2 = filler(inputs2)

    #Smooth
    inputs2 = weight_training_data(inputs2, 0.3)

    return inputs2

# Create outputs2
def create_outputs_2(state):
    statedata, datadic = gd.get_data(state)
    return statedata[factors3]

# Create prediction2
def create_prediction2(state):
    prediction1, weights1 = create_prediction1(state, factors1)
    inputs2 = create_inputs_2(prediction1, state)
    outputs2 = create_outputs_2(state)
    model2 = Ridge()

    model2.fit(inputs2[inputs2.index < 2005].values, outputs2.loc[outputs2.index < 2005].values)

    prediction2 = pd.DataFrame(model2.predict(inputs2), index=np.arange(1960, 2051, 1), columns=factors3)
    weights2 = pd.DataFrame(model2.predict(np.identity(len(inputs2.columns))), index=inputs2.columns, columns=factors3)

    graph_predictions(outputs2, prediction2, factors3, datadic, 'PredictiveGraphs/'+state+'Pred2.png', title='Renewable Production', state=state)

    return prediction1, prediction2, weights1, weights2

# Create prediction3
def create_prediction3(state):

    # Create prediction1, 2, to be used as inputdata
    prediction1, prediction2, weights1, weights2 = create_prediction2(state)
    prediction2[[factor for factor in factors1]] = prediction1[[factor for factor in factors1]]

    # Get outputs3
    statedata, datadic = gd.get_data(state)
    outputs3 = statedata[factors4]

    # Make the last model
    model3 = LinearRegression()
    model3.fit(prediction2[prediction2.index < 2005].values, outputs3.loc[outputs3.index < 2005].values)
    prediction3 = pd.DataFrame(model3.predict(prediction2.values), index=prediction2.index, columns=factors4)


    graph_predictions(outputs3, prediction3, [factors for factors in factors4], datadic, 'PredictiveGraphs/'+state+'Pred3.png', title='Nonrenewable Production', state=state)

    return prediction3, prediction2, prediction1

# do it all
def predict(state):
    prediction3, prediction2, prediction1 = create_prediction3(state)
    prediction3[prediction2.columns] = prediction2
    prediction3[prediction1.columns] = prediction1

    totals = pd.DataFrame(index=prediction3.index)
    totals['CLTCB'] = prediction3['CLTXB']
    totals['NGTCB'] = prediction3['NGTXB']
    totals['PATCB'] = prediction3['PATXB']
    totals['GETCB'] = prediction3['GEEGB']
    totals['HYTCB'] = prediction3['HYEGB']
    totals['NUTCB'] = prediction3['NUEGB']
    totals['SOTCB'] = prediction3['SOHCB']
    totals['WYTCB'] = prediction3['WYEGB']
    totals['EMTCB'] = np.zeros((len(totals)))

    emissions = co2.calc_sector_emissions(totals)

    fig, ax = plt.subplots()
    colors = plt.cm.jet(np.linspace(0, 1, len(emissions.columns)))
    for factor, color in zip(emissions.columns, colors):
        if factor[0:2] == 'EM':
            continue
        ax.plot(emissions[factor], color=color, label=final_source_dic[factor[0:2]])

    ax.set(title='Predicted Emissions in {} by Source'.format(state),
           xlabel='Years',
           ylabel='Megatons CO2')
    ax.legend()
    plt.savefig('PredictiveGraphs/'+state+'CO2.png')

    emissions = emissions.sum(axis=1)

    mortality = co2.calc_sector_mortality(totals)
    fig, ax = plt.subplots()
    colors = plt.cm.jet(np.linspace(0, 1, len(mortality.columns)))
    for factor, color in zip(mortality.columns, colors):
        if factor[0:2] == 'EM':
            continue
        ax.plot(mortality[factor], color=color, label=final_source_dic[factor[0:2]])

    ax.set(title='Predicted Deaths from Pollution in {} by Source'.format(state),
           xlabel='Years',
           ylabel='Annual Deaths')
    ax.legend()
    plt.savefig('PredictiveGraphs/'+state+'Mort.png')
    mortality = mortality.sum(axis=1)


    return emissions, mortality

statelist = ['AZ', 'CA', 'NM', 'TX']
alle = pd.DataFrame(index=np.arange(1960, 2051, 1), columns=statelist)
allm = pd.DataFrame(index=np.arange(1960, 2051, 1), columns=statelist)
for state in statelist:
    alle[state], allm[state] = predict(state)

fig, ax = plt.subplots()
for state in statelist:
    ax.plot(alle[state], label=state, alpha=0.5)
ax.set(title='Projected Emissions by State')
ax.set(xlabel='Years')
ax.set(ylabel='Megatons CO2 Equivilant')
ax.legend()
plt.savefig('PredictiveGraphs/AllCO2.png', bbox_inches='tight')

fig, ax = plt.subplots()
for state in statelist:
    ax.plot(alle[state], label=state, alpha=0.5)
ax.set(title='Projected Pollution Deaths by State')
ax.set(xlabel='Years')
ax.set(ylabel='Deaths (Annual)')
ax.legend()
plt.savefig('PredictiveGraphs/AllDeaths.png', bbox_inches='tight')

print(allm, alle)

exit()


## Globals
statedata, datadic= gd.get_data('AZ')

final_source_dic = {'CL':'Coal', 'NG': 'Natural Gas', 'PA': 'Petroleum Products',
             'NU': "Nuclear Electric Power", 'EM': 'Fuel Ethanol',
             'GE': 'Geothermal', 'HY': 'Hydroelectric', 'SO': 'Solar Thermal',
              'WY': 'Wind'}

final_sector_dic = {'CC': 'Comercial', 'IC': 'Industrial', 'RC':'Residential', 'AC': 'Transportation',
             'EI': 'Electric Sector Consumption', 'EG': 'Energy Sector Generation', 'HC':'Residential and Commercial'}

# Create layer1 factors
layer1_factors = ['Population',
                  'Q20', 'Q40', 'Q60', 'Q80', 'Solar', 'Wind', 'Coal', 'Natural Gas',
                  'Petroleum Products',	'Nuclear Electric Power', 'Oil Price']
# List of layer2_factors
layer2_factors = [factor for factor in statedata.columns if factor[0:2] in final_source_dic
                  and factor[2:4] in final_sector_dic and factor[-1] == 'B']

#for item in ['NUTCB', 'WYTCB', 'HYTCB', 'SOTCB']:
    #layer2_factors.append(item)

# Create inputdata
def create_inputdata(state):

    # Get inputdata
    inputdata = pd.DataFrame(index=np.arange(1960, 2051, 1), columns=layer1_factors)
    inputdata['Population'] = pop.get_all_pop_data(state)
    inputdata['Solar'] = renew.renewable_project('Solar')
    inputdata['Wind'] = renew.renewable_project('Wind')
    inputdata[['Q20', 'Q40', 'Q60', 'Q80']] = climate.predict_climate_data(state)
    inputdata[[ 'Coal', 'Natural Gas', 'Petroleum Products', 'Nuclear Electric Power', 'Oil Price']] = pd.read_csv(state +'_NonRen_Prices.csv', index_col=0, header=0)

    # Fill NANs with earliest observed value
    for factor in inputdata.columns:
        bool = inputdata.loc[:, factor].notnull()
        earliest = min(inputdata.index[bool])
        bool = [not i for i in bool]
        inputdata.at[bool, factor] = inputdata.loc[earliest, factor]

    return inputdata

# Create outputdata
def create_outputdata(state):
    statedata, datadic = gd.get_data(state)
    statedata = statedata[layer2_factors]
    #statedata = statedata.ewm(alpha=0.5).mean()
    return statedata, datadic

# exponentially weight in reverse training data
def weight_training_data(data, alpha):
    # reverse
    data = data.reindex(index=data.index[::-1])
    #weight
    data = data.ewm(alpha=alpha).mean()
    #reverse again
    data = data.reindex(index=data.index[::-1])
    return data


statecode = 'NM'

inputdata = create_inputdata(statecode)
statedata, datadic = gd.get_data(statecode)
outputdata, datadic = create_outputdata(statecode)


inputdata = weight_training_data(inputdata, 0.1)

model = Ridge()

model.fit(inputdata[(1995 < inputdata.index) & (inputdata.index < 2005)].values, outputdata.loc[(1995 < outputdata.index) & (outputdata.index < 2005)].values)

prediction = pd.DataFrame(model.predict(inputdata), index=np.arange(1960, 2051, 1), columns=layer2_factors)
weights = pd.DataFrame(model.predict(np.identity(len(inputdata.columns))), index=inputdata.columns, columns=layer2_factors)
