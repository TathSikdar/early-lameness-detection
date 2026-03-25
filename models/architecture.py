from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import os

class AnomalyDetector:
    """
    Unsupervised anomaly detection for lameness using Isolation Forest.
    """
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def fit(self, features):
        """
        Train on normal data (features from healthy cows).
        features: list of dicts or array of feature vectors.
        """
        if isinstance(features[0], dict):
            # Convert dicts to arrays
            X = np.array([[f['stride_length'], f['head_bobbing'], f['symmetry'], f['cadence']] for f in features])
        else:
            X = np.array(features)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
    
    def predict(self, features):
        """
        Predict anomaly score for new features.
        Returns score between 0 (normal) and 1 (anomalous).
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        if isinstance(features, dict):
            X = np.array([[features['stride_length'], features['head_bobbing'], features['symmetry'], features['cadence']]])
        else:
            X = np.array(features)
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.decision_function(X_scaled)
        # Convert to 0-1 scale, where -1 is normal, 1 is anomaly
        normalized_scores = (scores - scores.min()) / (scores.max() - scores.min()) if scores.max() > scores.min() else np.zeros(len(scores))
        return normalized_scores[0]
    
    def save_model(self, path):
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
    
    def load_model(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.is_trained = True