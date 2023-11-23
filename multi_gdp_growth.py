# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Load dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/javaih/OneDrive - Office for National Statistics/World bank analysis/final_demographics_data.csv')

# Drop rows with missing values for the selected variables
df_cleaned = df[['GDP growth (annual %)', 
                 'Population growth (annual %)', 
                 'Human capital index (HCI) (scale 0-1)', 
                 'Unemployment, total (% of total labor force) (modeled ILO estimate)',
                 'Military expenditure (% of GDP)']].dropna()

# Scale all independent variables to the range [0, 1]
scaler = MinMaxScaler()
df_cleaned[df_cleaned.columns[1:]] = scaler.fit_transform(df_cleaned[df_cleaned.columns[1:]])

# Select the dependent and independent variables
dependent_variable = df_cleaned['GDP growth (annual %)']
independent_variables = df_cleaned[['Population growth (annual %)', 
                                    'Human capital index (HCI) (scale 0-1)',
                                    'Unemployment, total (% of total labor force) (modeled ILO estimate)', 
                                    'Military expenditure (% of GDP)']]

# Add a constant term to the independent variables matrix
independent_variables = sm.add_constant(independent_variables)

# Create and fit the linear regression model
model = sm.OLS(dependent_variable, independent_variables)
results = model.fit()

# Print the regression results
print(results.summary())
