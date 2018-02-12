import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphinitialdata as gd
import smallindexes as si
import co2calc as co2

## Globals
statedata, datadic= gd.get_data('AZ')

final_source_dic = {'CL':'Coal', 'NG': 'Natural Gas', 'PA': 'Petroleum Products',
             'NU': "Nuclear Electric Power", 'EM': 'Fuel Ethanol',
             'GE': 'Geothermal', 'HY': 'Hydroelectric', 'SO': 'Solar Thermal',
              'WY': 'Wind'}

final_sector_dic = {'CC': 'Comercial', 'IC': 'Industrial', 'RC':'Residential', 'AC': 'Transportation',
             'EI': 'Electric Sector Consumption', 'EG': 'Energy Sector Generation', 'HC':'Residential and Commercial'}

# List of layer2_factors
layer2_factors = [factor for factor in statedata.columns if factor[0:2] in final_source_dic
                  and factor[2:4] in final_sector_dic and factor[-1] == 'B']
