main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample dataset
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)
scores = 40 + 6 * study_hours + np.random.normal(0, 5, 100)

# Create DataFrame
data = pd.DataFrame({
    'study_hours': study_hours,
    'score': scores
})

# Split data
X = data[['study_hours']]
y = data['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Coefficient: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Prediction
hours = 5
predicted_score = model.predict([[hours]])[0]
print(f"\nPredicted score for {hours} hours: {predicted_score:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2, label='Prediction Line')
plt.xlabel('Study Hours')
plt.ylabel('Score')
plt.title('Study Hours vs Score Prediction')
plt.legend()
plt.grid(True)
plt.savefig('prediction_plot.png')
plt.show()
