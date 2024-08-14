import pandas as pd
import numpy as np
from collections import Counter
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        most_common = Counter(k_nearest_labels).most_common(1)
        logging.debug(f"K nearest labels: {k_nearest_labels}, Most common: {most_common}")
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
    if len(y_true) == 0 or len(y_pred) == 0:
        logging.error(f"Empty array in accuracy calculation. y_true: {len(y_true)}, y_pred: {len(y_pred)}")
        return np.nan
    accuracy = np.mean(y_true == y_pred)
    logging.debug(f"Accuracy calculation - y_true: {y_true[:5]}, y_pred: {y_pred[:5]}")
    logging.debug(f"Calculated accuracy: {accuracy}")
    if np.isnan(accuracy):
        logging.warning(f"NaN accuracy detected. y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
        logging.warning(f"y_true sample: {y_true[:5]}, y_pred sample: {y_pred[:5]}")
    return accuracy

# Load and prepare data
data = pd.read_csv('pokemon.csv')
features = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense']
target_column = 'type1'

X = data[features].values
y = data[target_column].values

logging.info(f"Dataset shape: {X.shape}")
logging.info(f"Features: {features}")
logging.info(f"Target column: {target_column}")

# Encode target values
target_encoder = {value: index for index, value in enumerate(np.unique(y))}
y_encoded = np.array([target_encoder[value] for value in y])

logging.info(f"Unique target values: {list(target_encoder.keys())}")

# Initialize KFold
n_splits = 10
kf = custom_kfold(n_splits, X.shape[0])

# Test different k values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    accuracies = []
    accuracies_scaled = []
    
    logging.info(f"\nStarting calculations for k = {k}")
    
    for fold, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
        
        logging.debug(f"Fold {fold + 1}, Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Unscaled data
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        logging.debug(f"Unscaled - y_pred shape: {y_pred.shape}, y_test shape: {y_test.shape}")
        accuracy = calculate_accuracy(y_test, y_pred)
        accuracies.append(accuracy)
        
        # Scaled data
        X_train_scaled = custom_scale(X_train)
        X_test_scaled = custom_scale(X_test)
        
        knn_scaled = KNN(k=k)
        knn_scaled.fit(X_train_scaled, y_train)
        y_pred_scaled = knn_scaled.predict(X_test_scaled)
        logging.debug(f"Scaled - y_pred shape: {y_pred_scaled.shape}, y_test shape: {y_test.shape}")
        accuracy_scaled = calculate_accuracy(y_test, y_pred_scaled)
        accuracies_scaled.append(accuracy_scaled)
    
    logging.info(f"k = {k}")
    logging.info(f"Average accuracy (unscaled): {np.mean(accuracies):.4f}")
    logging.info(f"Average accuracy (scaled): {np.mean(accuracies_scaled):.4f}")

# Example predictions
logging.info("\nExample predictions:")
knn = KNN(k=5)
knn.fit(X, y_encoded)
example_pokemon = X[:5]
predictions = knn.predict(example_pokemon)
for i, pred in enumerate(predictions):
    logging.info(f"Pokemon {i+1}: Predicted type: {list(target_encoder.keys())[list(target_encoder.values()).index(pred)]}, Actual type: {y[i]}")