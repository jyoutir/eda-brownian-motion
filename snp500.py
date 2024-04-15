import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


# Load the uploaded CSV file (HAVE TO CHANGE PATH IF YOU WANT TO RUN ON OWN COMPUTER)
file_path = '/Users/Library/Mobile Documents/com~apple~CloudDocs/SOR project/SandP500.csv'  
data = pd.read_csv(file_path)

# Converts the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Filter data between 1st January 2023 and 1st November 2023
filtered_data = data[(data['Date'] >= '2023-01-01') & (data['Date'] <= '2023-07-31')].copy()

# Calculate log returns
filtered_data['Log Returns'] = np.log(filtered_data['S&P500'] / filtered_data['S&P500'].shift(1))

# Remove the NaN value generated by the shift operation
log_returns = filtered_data['Log Returns'].dropna()




from statsmodels.tsa.stattools import adfuller

adf_result = adfuller(log_returns.dropna())

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
shapiro_test_result = stats.shapiro(log_returns)
shapiro_test_statistic, shapiro_test_pvalue = shapiro_test_result

# Print the results
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic}, p-value: {shapiro_test_pvalue}")

# Interpret the results
if shapiro_test_pvalue > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")
    
# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(log_returns, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of S&P 500 Log Returns (Jan 2023 - July 2023)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()



