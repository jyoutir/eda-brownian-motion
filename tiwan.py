import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the Taiwan Weighted dataset
taiwan_weighted_data = pd.read_excel('/Users/jyoutirraj/Library/Mobile Documents/com~apple~CloudDocs/SOR project/Taiwan-Weighted.xlsx')  # Update this to the path of your file

# Convert 'Date' column to datetime format
taiwan_weighted_data['Date'] = pd.to_datetime(taiwan_weighted_data['Date'])

# Filter data between September 2019 and March 2020
filtered_taiwan_weighted_data = taiwan_weighted_data[(taiwan_weighted_data['Date'] >= '2019-09-01') & (taiwan_weighted_data['Date'] <= '2020-03-31')].copy()

# Calculate log returns for the 'Close' column
filtered_taiwan_weighted_data['Log Returns'] = np.log(filtered_taiwan_weighted_data['Close'] / filtered_taiwan_weighted_data['Close'].shift(1))

# Perform the Shapiro-Wilk test for normality on log returns
shapiro_test_result_taiwan_weighted = stats.shapiro(filtered_taiwan_weighted_data['Log Returns'].dropna())  # Dropping NaN values
shapiro_test_statistic_taiwan_weighted, shapiro_test_pvalue_taiwan_weighted = shapiro_test_result_taiwan_weighted

# Print the Shapiro-Wilk test results with more decimal places
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic_taiwan_weighted}, p-value: {shapiro_test_pvalue_taiwan_weighted}")

# Interpret the results
if shapiro_test_pvalue_taiwan_weighted > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")

# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(filtered_taiwan_weighted_data['Log Returns'].dropna(), bins=30, edgecolor='k', alpha=0.7)  
plt.title('Histogram of Taiwan Weighted Log Returns (Sept 2019 - Mar 2020)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()

# Generate and show the Q-Q plot for log returns
plt.figure(figsize=(8, 6))
stats.probplot(filtered_taiwan_weighted_data['Log Returns'].dropna(), dist="norm", plot=plt)  # Dropping NaN values
plt.title('Q-Q Plot of Taiwan Log Returns (Sep 2019 - March 2020)')
plt.show()
