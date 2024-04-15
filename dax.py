import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the DAX dataset
dax_data = pd.read_csv('/Users/jyoutirraj/Library/Mobile Documents/com~apple~CloudDocs/SOR project/DAX.csv')  # Update this to the path of your DAX file

# Convert 'Date' column to datetime format
dax_data['Date'] = pd.to_datetime(dax_data['Date'], format='%d/%m/%Y')

# Filter data between June 2017 and December 2017
filtered_dax_data = dax_data[(dax_data['Date'] >= '2017-06-01') & (dax_data['Date'] <= '2017-12-31')].copy()

# Calculate log returns for the 'Close' column
filtered_dax_data['Log Returns'] = np.log(filtered_dax_data['Close'] / filtered_dax_data['Close'].shift(1))



from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(filtered_dax_data.dropna())

adf_statistic = adf_result[0]
adf_pvalue = adf_result[1]
critical_values = adf_result[4]

print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {adf_pvalue}')
print('Critical Values:')
for key, value in critical_values.items():
    print(f'\t{key}: {value}')

# Interpret the results
if adf_pvalue < 0.05:
    print("ADF Test: The log returns data does not have a unit root and is stationary (reject H0).")
else:
    print("ADF Test: The log returns data has a unit root and is non-stationary (fail to reject H0).")







# Perform the Shapiro-Wilk test for normality on log returns
shapiro_test_result_dax = stats.shapiro(filtered_dax_data['Log Returns'].dropna())  # Make sure to drop NaN values
shapiro_test_statistic_dax, shapiro_test_pvalue_dax = shapiro_test_result_dax

# Print the Shapiro-Wilk test results with more decimal places
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic_dax}, p-value: {shapiro_test_pvalue_dax}")

# Interpret the results
if shapiro_test_pvalue_dax > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")

# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(filtered_dax_data['Log Returns'].dropna(), bins=30, edgecolor='k', alpha=0.7)  # Make sure to drop NaN values
plt.title('Histogram of DAX Log Returns (June 2017 - December 2017)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()




