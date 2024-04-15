import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the China A50 dataset
china_a50_data = pd.read_excel('/Users/Library/Mobile Documents/com~apple~CloudDocs/SOR project/China-A50.xlsx')  # Update this to the path of your China A50 file

# Convert 'Date' column to datetime format
china_a50_data['Date'] = pd.to_datetime(china_a50_data['Date'])

# Filter data between August 2017 and February 2018
filtered_china_a50_data = china_a50_data[(china_a50_data['Date'] >= '2017-08-01') & (china_a50_data['Date'] <= '2018-02-28')].copy()

# Calculate log returns for the 'Close' column
filtered_china_a50_data['Log Returns'] = np.log(filtered_china_a50_data['Close'] / filtered_china_a50_data['Close'].shift(1))


from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(filtered_china_a50_data.dropna())

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
shapiro_test_result_china_a50 = stats.shapiro(filtered_china_a50_data['Log Returns'].dropna())  # Dropping NaN values
shapiro_test_statistic_china_a50, shapiro_test_pvalue_china_a50 = shapiro_test_result_china_a50

# Print the Shapiro-Wilk test results with more decimal places
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic_china_a50}, p-value: {shapiro_test_pvalue_china_a50}")

# Interpret the results
if shapiro_test_pvalue_china_a50 > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")

# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(filtered_china_a50_data['Log Returns'].dropna(), bins=10, edgecolor='k', alpha=0.7)  # Dropping NaN values
plt.title('Histogram of China A50 Log Returns (August 2017 - February 2018)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()

# Generate and show the Q-Q plot for log returns
plt.figure(figsize=(8, 6))
stats.probplot(filtered_china_a50_data['Log Returns'].dropna(), dist="norm", plot=plt)  # Dropping NaN values
plt.title('Q-Q Plot of China A50 Log Returns (August 2017 - February 2018)')
plt.show()
