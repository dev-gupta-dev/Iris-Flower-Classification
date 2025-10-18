import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load built-in Iris dataset
iris = load_iris()

# Convert to DataFrame for readability
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("📂 First 5 rows of dataset:")
print(df.head())
X = iris.data        # Features
y = iris.target      # Labels (0=Setosa, 1=Versicolor, 2=Virginica)

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)  # k=3
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)

# Accuracy
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# Detailed report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))
# Example: sepal length=5.1, sepal width=3.5, petal length=1.4, petal width=0.2
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
prediction = model.predict(sample)
print("\n🌸 Predicted species:", iris.target_names[prediction][0])
