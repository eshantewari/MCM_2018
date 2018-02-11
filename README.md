# MCM_2018

## Contest Information
Team Number: 76796
Problem C

## Initial Formatting Docs:
formatinitialdata will structure the data better (it will also split the file into four files, one for each state)
graphinitialdata creates a data dictionary and graphs specific attributes within and across states

## Understanding EIA Data

The ESA technical notes are here: https://www.eia.gov/state/seds/seds-technical-notes-complete.php

### Understanding MSN Codes (copied from EIA website):
The MSNs are five-character codes, most of which are structured as follows:
First and second characters - describes an energy source (for example, NG for natural gas, MG for motor gasoline)
Third and fourth characters - describes an energy sector or an energy activity (for example, RC for residential consumption, PR for production)

Fifth character - describes a type of data (for example, P for data in physical unit, B for data in billion Btu)

For more, see https://www.eia.gov/state/seds/sep_use/notes/use_guide.pdf

#### Last character
The fifth character of the variable names in SEDS identifies the type of data by
using one of the following letters:
B = data in British thermal units (Btu)
K = factor for converting data from physical units to Btu
M = data in alternative physical units
P = data in standardized physical units
S = share or ratio expressed as a fraction
V = value in million dollars

### Total end-use consumption
"A new set of tables for total end-use energy consumption, price, and expenditure estimates is introduced in this cycle. Estimates for total end-use consumption and expenditures are calculated by summing the consumption and expenditures, respectively, of the four end-use sectors: residential, commercial, industrial, and transportation. Estimates for total end-use prices are calculated by dividing total end-use expenditures by the sum of all end-use consumption with prices associated with them."
