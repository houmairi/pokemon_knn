import pandas as pd
import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common()
        
        if not most_common:
            return self.y_train[np.argmin(distances)]  # Return the label of the single nearest neighbor
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            return most_common[0][0]  # Clear winner
        else:
            # Tie-breaking: choose the label of the nearest neighbor among tied classes
            tied_labels = [label for label, count in most_common if count == most_common[0][1]]
            for i in k_indices:
                if self.y_train[i] in tied_labels:
                    return self.y_train[i]
    
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0] if most_common else None

def custom_kfold(n_splits, n_samples):
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        yield indices[start:stop], np.concatenate([indices[:start], indices[stop:]])
        current = stop

def custom_scale(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero
    return (X - mean) / std

def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load the Pokemon dataset
data = pd.read_csv('pokemon.csv')

print("Dataset shape:", data.shape)
print("\nSample of the first few rows:")
print(data.head())

# Select features and target
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense']
target_column = 'type1'

print(f"\nUsing features: {features}")
print(f"Using target column: {target_column}")

# Check if all required columns are present
missing_columns = [col for col in features + [target_column] if col not in data.columns]
if missing_columns:
    raise ValueError(f"Missing columns in the dataset: {missing_columns}")

X = data[features].values
y = data[target_column].values

print("\nFeature statistics:")
print(pd.DataFrame(X, columns=features).describe())

print("\nUnique target values:", np.unique(y))

# Encode target values
target_encoder = {value: index for index, value in enumerate(np.unique(y))}
target_decoder = {index: value for value, index in target_encoder.items()}
y_encoded = np.array([target_encoder[value] for value in y])

print("\nTarget encoding:")
for value, index in target_encoder.items():
    print(f"{value}: {index}")

# Initialize KFold
n_splits = 10
kf = custom_kfold(n_splits, X.shape[0])

# Test different k values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    accuracies = []
    accuracies_scaled = []
    
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        # Unscaled data
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_pred)
        accuracies.append(accuracy)
        
        # Scaled data
        X_train_scaled = custom_scale(X_train)
        X_test_scaled = custom_scale(X_test)
        
        knn_scaled = KNN(k=k)
        knn_scaled.fit(X_train_scaled, y_train)
        y_pred_scaled = knn_scaled.predict(X_test_scaled)
        accuracy_scaled = calculate_accuracy(y_test, y_pred_scaled)
        accuracies_scaled.append(accuracy_scaled)
    
    print(f"\nk = {k}")
    print(f"Average accuracy (unscaled): {np.mean(accuracies):.4f}")
    print(f"Average accuracy (scaled): {np.mean(accuracies_scaled):.4f}")

# Example predictions
print("\nExample predictions:")
knn = KNN(k=5)
knn.fit(X, y_encoded)
example_pokemon = X[:5]
predictions = knn.predict(example_pokemon)
for i, pred in enumerate(predictions):
    print(f"Pokemon {i+1}: Predicted type: {target_decoder[pred]}, Actual type: {y[i]}")