import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the FTSE 100 dataset
ftse100_data = pd.read_excel('/Users/Library/Mobile Documents/com~apple~CloudDocs/SOR project/FTSE100.xlsx')  # Update this to the path of your FTSE 100 file

# Convert 'Date' column to datetime format
ftse100_data['Date'] = pd.to_datetime(ftse100_data['Date'])

# Filter data between June 2017 and December 2017
filtered_ftse100_data = ftse100_data[(ftse100_data['Date'] >= '2017-06-01') & (ftse100_data['Date'] <= '2017-12-31')].copy()

# Calculate log returns for the 'Close' column
filtered_ftse100_data['Log Returns'] = np.log(filtered_ftse100_data['Close'] / filtered_ftse100_data['Close'].shift(1))

# Remove the NaN value generated by the shift operation
log_returns_ftse100 = filtered_ftse100_data['Log Returns'].dropna()




from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(log_returns_ftse100.dropna())

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
shapiro_test_result_ftse100 = stats.shapiro(log_returns_ftse100)
shapiro_test_statistic_ftse100, shapiro_test_pvalue_ftse100 = shapiro_test_result_ftse100


# Print the Shapiro-Wilk test results with more decimal places
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic_ftse100}, p-value: {shapiro_test_pvalue_ftse100}")

# Interpret the results
if shapiro_test_pvalue_ftse100 > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")

# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(log_returns_ftse100, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of FTSE 100 Log Returns (June 2017 - December 2017)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()


