import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("Housing.csv")

# Convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)



# Create new input as list
new_house = [[
    3000,  # area
    3,     # bedrooms
    2,     # bathrooms
    2,     # stories
    1,     # mainroad_yes
    0,     # guestroom_yes
    0,     # basement_yes
    0,     # hotwaterheating_yes
    1,     # airconditioning_yes
    2,     # parking
    1,     # prefarea_yes
    0,     # semi-furnished
    1      # unfurnished
]]

new_house_df = pd.DataFrame(new_house, columns=X.columns)

predicted_price = model.predict(new_house_df)

print("Predicted Price:", predicted_price[0])


