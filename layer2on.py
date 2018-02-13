import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import smallindexes as si
import co2calc as co2
from layer1 import *

final_sector_dic['TC'] = 'Total Consumption'

axisdic = {'B': 'BTU (Billions)', 'D':'Annual Pollution Deaths', 'T': 'Megatons of CO2 Equivalent',
           'V':'Cost of Emissions (Millions of Dollars)'}

### This file takes the second layer and turns it into mortality and CO2 emissions outcomes, and then also graphs them

## Globals, including layer3 factors
layer3_factors = [item + 'TCB' for item in final_source_dic]
if 'NUTCB' in layer3_factors:
    layer3_factors.remove('NUTCB')
    layer3_factors.append('NUEGB')

#layer2_factors should be the columns
#some years should be the index

def create_totals(layer2_forecasts):
    totals = pd.DataFrame(index=layer2_forecasts.index, columns=layer3_factors)
    for factor in totals.columns:
        sumands = [item for item in layer2_forecasts.columns if factor[0:2] == item[0:2]]
        totals[factor] = layer2_forecasts[sumands].sum(axis=1)

    return totals

# Get the prediction
layer3_forecasts = create_totals(prediction)


# Pop adj will NOT work yet.

# Get emissions, mortality data
emissions = co2.calc_sector_emissions(layer3_forecasts)
mortality = co2.calc_sector_mortality(layer3_forecasts)
#co2.graph_emissions_profile_v2(emissions, datadic, 'TX', unit='T', min=0.01, pop_adj=False, inf_adj=False, sum=False)