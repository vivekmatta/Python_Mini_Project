import pandas
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Reading and cleaning the bicycle count data
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge'] = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',', '', regex=True))
dataset_2['Manhattan Bridge'] = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',', '', regex=True))
dataset_2['Queensboro Bridge'] = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',', '', regex=True))
dataset_2['Williamsburg Bridge'] = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',', '', regex=True))

# Reading and cleaning the behavior-performance data
with open('behavior-performance.txt') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:], columns=raw_data[0])

df = df.apply(pandas.to_numeric, errors='ignore')

# Part 1 - Check performance with all combinations of sensor positions (4 possibilities)
sensor_1 = dataset_2['Brooklyn Bridge'].to_numpy()
sensor_2 = dataset_2['Manhattan Bridge'].to_numpy()
sensor_3 = dataset_2['Williamsburg Bridge'].to_numpy()
sensor_4 = dataset_2['Queensboro Bridge'].to_numpy()

all_traffic = dataset_2[['Queensboro Bridge', 'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge']].to_numpy()
avg_traffic = all_traffic.mean(axis=1)

errors = []
for sensor in [sensor_1, sensor_2, sensor_3, sensor_4]:
    errors.append(np.mean(np.abs(sensor - avg_traffic) / avg_traffic))

# Part 2 - Supervised Clustering

# Ensure the 'Precipitation' column is treated as a string
dataset_2['Precipitation'] = dataset_2['Precipitation'].astype(str)

# Handling the 'Precipitation' column
stripped_prec = pandas.to_numeric(
    dataset_2['Precipitation'].str.split(' ', expand=True)[0]
    .replace('T', '0.001')
    .replace('', '0')
)

# Concatenating features
features = pandas.concat([dataset_2[['High Temp (°F)', 'Low Temp (°F)']], stripped_prec], axis=1)

# Checking the shapes of features and traffic
print(features.shape, avg_traffic.shape)

# Linear Regression
linreg = LinearRegression()
linreg.fit(features, avg_traffic)
op = linreg.predict(features)

# Plotting
plt.scatter(range(len(op)), op, color='red')
plt.scatter(range(len(avg_traffic)), avg_traffic, color='blue')
plt.grid()
plt.show()
plt.savefig("Problem.png")
