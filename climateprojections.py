import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import graphinitialdata as gd

# Clean initial climate data
def clean_climate_data(state):
    # Get initial data
    path = state + 'ClimateData.txt'
    rough_data = pd.read_csv(path, header=0, index_col=0)
    clean_data = pd.DataFrame(index=np.arange(1960, 2010,1), columns=['Q20','Q40','Q60', 'Q80'])

    # Clean data
    for year in clean_data.index:

        # Get all relevant dates
        dates = [factor for factor in rough_data.index if int(str(factor)[0:-2]) == year]

        # Get quartiles
        quartiles = np.percentile(rough_data.loc[dates, 'Value'], [20, 40, 60, 80])

        clean_data.loc[year] = np.array(quartiles)

    return clean_data

# Predict climate data
def predict_climate_data(state):

    # Get cleaned data
    observed_data = clean_climate_data(state)

    # Years
    years = np.arange(1960, 2051, 1)

    # Create model dataframe
    modeled_data = pd.DataFrame(index=np.arange(1960, 2051, 1), columns=observed_data.columns)

    fig, ax = plt.subplots()

    # Model each quartile  from its past self
    for q in observed_data.columns:

        # Fit model - will be quadratic
        values = pd.Series(np.array(observed_data[q].values, dtype=float)).ewm(alpha=0.25).mean()
        fit = np.polyfit(observed_data.index, values, 2, w=np.sqrt(values))

        # Create forecast
        forecast = fit[0]*(years**2) + fit[1]*years + fit[2]
        forecast = pd.Series(forecast, years)

        # Plot if it's the main thing
        if __name__ == '__main__':

            line1 = ax.plot(observed_data[q], color='blue', alpha=0.8)
            line2 = ax.plot(observed_data.index, values, color='yellow', alpha=0.8)
            line3 = ax.plot(forecast, color='red', linestyle='dotted', alpha=0.8)
            ax.set(title='Climate Projections for {} in Various Quartiles'.format(state))
            ax.set(xlabel='Year', ylabel='Temperature (F)')

        modeled_data[q] = forecast

    # Plot and save graph
    legend = ax.legend(['Observed Climate Data', 'Smoothed Climate Data', 'Projected Climate Data'])
    path = 'ClimateGraphs/'+ state + 'ClimateGraph' + '.png'
    if __name__ == '__main__':
        plt.savefig(path, bbox_inches='tight')

    # Create final prediction/data mesh to be returned
    modeled_data = modeled_data.loc[modeled_data.index > 2009]

    modeled_data = observed_data.append(modeled_data)
    return modeled_data
