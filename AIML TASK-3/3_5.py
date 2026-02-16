# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")

# Use only one feature for visualization
X = df[['area']]
y = df['price']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Plot scatter plot
plt.scatter(X, y)
plt.plot(X, y_pred)  # regression line

plt.title("Regression Line: Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")

plt.show()
