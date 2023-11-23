# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler

# Load dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/javaih/OneDrive - Office for National Statistics/World bank analysis/final_demographics_data.csv')

# Drop rows with missing values for specific columns
df_cleaned = df[['Women Business and the Law Index Score (scale 1-100)',
                 'Individuals using the Internet (% of population)',]].dropna()

# Scale all variables to the range [0, 1]
scaler = MinMaxScaler()
df_cleaned[df_cleaned.columns[1:]] = scaler.fit_transform(df_cleaned[df_cleaned.columns[1:]])

# Select the dependent variable
dependent_variable = df_cleaned['Women Business and the Law Index Score (scale 1-100)']

# Select the independent variable(s)
independent_variables = df_cleaned[['Individuals using the Internet (% of population)']]

# Add a constant term to the independent variables matrix
independent_variables = sm.add_constant(independent_variables)

# Fit the Ordinary Least Squares (OLS) regression model
model = sm.OLS(dependent_variable, independent_variables)
results = model.fit()

# Display the regression results
print(results.summary())
