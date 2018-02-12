import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import smallindexes as si
import co2calc as co2
from layer1 import *

### This file takes the second layer and turns it into mortality and CO2 emissions outcomes, and then also graphs them

## Globals, including layer3 factors
layer3_factors = [item + 'TCB' for item in final_source_dic]
print(layer3_factors)

#layer2_factors should be the columns
#some years should be the index

def create_totals(layer2_forecasts):
    totals = pd.DataFrame(index=layer2_forecasts.index, columns=layer3_factors)
    for factor in totals.columns:
        sumands = [item for item in layer2_forecasts.columns if factor[0:2] == item[0:2]]
        totals[factor] = layer2_forecasts[sumands].sum(axis=1)

    return totals

index=np.arange(2010, 2051, 1)
sample_data = pd.DataFrame(data=np.random.standard_exponential((len(index),len(layer2_factors))), index=index , columns=layer2_factors)
layer3_forecasts = create_totals(sample_data)

# Reminder - when available, add in population data
si.graph_source_profile('PA', sample_data, datadic, -1, 'TX', pop_adj = 'False', inf_adj = 'False')
emissions = co2.calc_sector_emissions(layer3_forecasts)
mortality = co2.calc_sector_mortality(layer3_forecasts)

# Pop adj will NOT work yet.
co2.graph_emissions_profile_v2(emissions, datadic, 'TX', unit='T', min=0.01, pop_adj=True, inf_adj=False, sum=False)