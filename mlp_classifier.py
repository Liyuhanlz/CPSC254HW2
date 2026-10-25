from sklearn.datasets import load_iris # 1. Load the Iris Flower dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into 80/20 training/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train the decision tree algorithm using MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=3, random_state=42)
mlp.fit(X_train, y_train)
train_accuracy = accuracy_score(y_train, mlp.predict(X_train)) * 100
test_accuracy = accuracy_score(y_test, mlp.predict(X_test)) * 100
print()
print("Accuracy with simple MLPClassifier")
print(f"Baseline: Train: {train_accuracy:.2f}%, Test: {test_accuracy:.2f}%")
print()

# 4. Adjust or change any MLP hyperparameters
mlp2 = MLPClassifier(hidden_layer_sizes=3, learning_rate_init=0.01, batch_size=16, max_iter=1000, random_state=42)
mlp2.fit(X_train, y_train)
print("Accuracy with adjusted MLP hyperparameters")
train_accuracy2 = accuracy_score(y_train, mlp2.predict(X_train)) * 100
test_accuracy2 = accuracy_score(y_test, mlp2.predict(X_test)) * 100
print(f"Adjusted: Train: {train_accuracy2:.2f}%, Test: {test_accuracy2:.2f}%")
print()

# 5. Increase layers and neurons until improved accuracy
inc_layers = [(5,), (10,), (20,), (5, 5), (10, 10), (20, 20), (5, 5, 5), (10, 10, 10), (20, 20, 20)]
print("Increasing Layers & Neurons")
for layers in inc_layers:
    model = MLPClassifier(hidden_layer_sizes=layers, learning_rate_init=0.01, batch_size=16, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    train_accuracy = accuracy_score(y_train, model.predict(X_train)) * 100
    test_accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
    print(f"Hidden Layers: {layers}: Train: {train_accuracy:.2f}%, Test: {test_accuracy:.2f}%")
print()