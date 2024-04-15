import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Load the DJ Shanghai dataset
dj_shanghai_data = pd.read_excel('/Users/jyoutirraj/Library/Mobile Documents/com~apple~CloudDocs/SOR project/DJ-Shanghai.xlsx')  

# Convert 'Date' column to datetime format
dj_shanghai_data['Date'] = pd.to_datetime(dj_shanghai_data['Date'])

# Filter data between January 2018 and July 2018
filtered_dj_shanghai_data = dj_shanghai_data[(dj_shanghai_data['Date'] >= '2018-01-01') & (dj_shanghai_data['Date'] <= '2018-07-31')].copy()

# Calculate log returns for the 'Close' column
filtered_dj_shanghai_data['Log Returns'] = np.log(filtered_dj_shanghai_data['Close'] / filtered_dj_shanghai_data['Close'].shift(1))

# Perform the Shapiro-Wilk test for normality on log returns
shapiro_test_result_dj_shanghai = stats.shapiro(filtered_dj_shanghai_data['Log Returns'].dropna())  # Dropping NaN values
shapiro_test_statistic_dj_shanghai, shapiro_test_pvalue_dj_shanghai = shapiro_test_result_dj_shanghai

# Print the Shapiro-Wilk test results with more decimal places
print(f"Shapiro-Wilk test statistic: {shapiro_test_statistic_dj_shanghai}, p-value: {shapiro_test_pvalue_dj_shanghai}")

# Interpret the results
if shapiro_test_pvalue_dj_shanghai > 0.05:
    print("The log returns data appears to be normally distributed (fail to reject H0).")
else:
    print("The log returns data does not appear to be normally distributed (reject H0).")

# Generate and show the histogram for log returns
plt.figure(figsize=(8, 6))
plt.hist(filtered_dj_shanghai_data['Log Returns'].dropna(), bins=30, edgecolor='k', alpha=0.7)  # Dropping NaN values
plt.title('Histogram of DJ Shanghai Log Returns (Sep 2019 - March 2020)') 
plt.xlabel('Log Return')
plt.ylabel('Frequency')
plt.show()


# Generate and show the Q-Q plot for log returns
plt.figure(figsize=(8, 6))
stats.probplot(filtered_dj_shanghai_data['Log Returns'].dropna(), dist="norm", plot=plt)  # Dropping NaN values
plt.title('Q-Q Plot of DJ Shanghai Log Returns (Jan 2018 - July 2018)')
plt.show()