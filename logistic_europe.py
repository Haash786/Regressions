# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('C:/Users/javaih/OneDrive - Office for National Statistics/World bank analysis/final_demographics_data.csv')

# Select relevant columns and drop missing values
df_cleaned = df[['Region', 'Urban population', 'Population, total', 'GDP per capita (constant 2010 US$)']].dropna()

# Calculate percentage of urban population
df_cleaned['Percentage Urban Population'] = (df_cleaned['Urban population'] / df_cleaned['Population, total']) * 100

# Create binary variable for Western Europe and North America
european_regions = ['EasternEurope', 'NorthernEurope', 'SouthernEurope', 'WesternEurope']
df_cleaned['Region_binary'] = df_cleaned['Region'].apply(lambda x: 1 if x in european_regions else 0)

# Select dependent and independent variables
independent = df_cleaned[['Percentage Urban Population', 'GDP per capita (constant 2010 US$)']]
dependent = df_cleaned['Region_binary']

# Scale independent variables
scaler = MinMaxScaler()
independent_scaled = scaler.fit_transform(independent)

# Split data into training and testing sets
independent_train, independent_test, dependent_train, dependent_test = train_test_split(
    independent_scaled, dependent, test_size=0.2, random_state=42)

# Create and fit logistic regression model
logit_model = LogisticRegression(penalty='l2', random_state=42)
logit_model.fit(independent_train, dependent_train)

# Make predictions on the test set
dependent_pred = logit_model.predict(independent_test)

# Display results
print("Classification Report:")
print(classification_report(dependent_test, dependent_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(dependent_test, dependent_pred))