import pandas as pd
from sklearn.linear_model import LinearRegression

import os
cwd = os.getcwd()
cwd

# Assign the dataframe to this variable.
# Load the data
bmi_life_data = pd.read_csv("01-regression/bmi_and_life_expectancy.csv")
output = bmi_life_data[['Life expectancy']]
input = bmi_life_data[['BMI']]

# Make and fit the linear regression model
# Fit the model and Assign it to bmi_life_model
bmi_life_model = LinearRegression()
bmi_life_model.fit(input,output)

# Mak a prediction using the model
# Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict(21.07931)
print("life expectancy for BMI (21.07931) is " + str(laos_life_exp))

from sklearn.preprocessing import StandardScaler, LabelEncoder
le = LabelEncoder()
bmi_life_data['Country'] = le.fit_transform(bmi_life_data['Country'])

input = bmi_life_data[['BMI', 'Country']]
bmi_life_model = LinearRegression()
bmi_life_model.fit(input,output)

test = pd.DataFrame([21.07931,10])
laos_life_exp2 = bmi_life_model.predict([[21.07931,10]])
print("life expectancy for BMI (21.07931) is " + str(laos_life_exp2))
