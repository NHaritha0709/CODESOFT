import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
# Load sample dataset (Advertising.csv)
# You can also use your custom CSV with similar structure
data = pd.read_csv(r"C:\Users\harit\OneDrive\intern\advertising.csv")

# Display first few rows
print(data.head())

# Define features and target
X = data[['TV', 'Radio', 'Newspaper']]  # Features
y = data['Sales']                       # Target

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Model evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plotting predictions vs actual
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()                                                                       
corr = data.corr()

# Plot heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()                                                                        
plt.figure(figsize=(8,5))
plt.plot(data['TV'], data['Sales'], label='TV', marker='o')
plt.plot(data['Radio'], data['Sales'], label='Radio', marker='o')
plt.plot(data['Newspaper'], data['Sales'], label='Newspaper', marker='o')
plt.title('Advertising Spend vs Sales')
plt.xlabel('Advertising Spend')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()
