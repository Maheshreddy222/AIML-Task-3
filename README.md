1️⃣ Import and Preprocess Dataset
Objective

Load the dataset and convert it into a format suitable for machine learning.

Steps performed

Imported dataset using pandas

Checked dataset structure

Converted categorical features into numerical using one-hot encoding

Prepared dataset for training

Code
import pandas as pd

df = pd.read_csv("Housing.csv")

# Convert categorical columns to numerical
df = pd.get_dummies(df, drop_first=True)

print(df.head())
Result

Dataset loaded successfully

All features converted to numeric format

Dataset ready for training

2️⃣ Split Data into Train-Test Sets
Objective

Split dataset into training and testing data.

Training data → teaches model
Testing data → evaluates model

Code
from sklearn.model_selection import train_test_split

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
Result

80% data used for training

20% data used for testing

3️⃣ Fit Linear Regression Model
Objective

Train Linear Regression model using training data.

Code
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully")
Result

Model learned relationship between features and price

Model ready to predict new values

4️⃣ Evaluate Model using MAE, MSE, and R²
Objective

Measure model performance using evaluation metrics.

Code
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)
Metric Interpretation

MAE (Mean Absolute Error)

Average prediction error

Lower is better

MSE (Mean Squared Error)

Squared prediction error

Lower is better

R² Score

Measures model accuracy

Range: 0 to 1

Higher is better

5️⃣ Plot Regression Line and Interpret Coefficients
Objective

Visualize regression line and understand model coefficients.

Code
import matplotlib.pyplot as plt

# Example using area feature
X_area = df[['area']]
y_price = df['price']

model.fit(X_area, y_price)

y_pred = model.predict(X_area)

plt.scatter(X_area, y_price)
plt.plot(X_area, y_pred)

plt.title("Regression Line: Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")

plt.show()

print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
Interpretation

Regression equation:

price = intercept + coefficient × area

Coefficient meaning:

Shows how much price increases when area increases by 1 unit

Intercept meaning:

Base price when area is zero
