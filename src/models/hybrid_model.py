import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Callable, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from utils.logging_config import get_logger

class HybridAnomalyDetector:
    def __init__(self, 
                 contamination: float = 0.1,
                 n_estimators_iso_forest: int = 100,
                 max_features_iso_forest: float = 1.0,
                 n_estimators_rf: int = 100,
                 max_depth_rf: Optional[int] = 10,
                 min_samples_split_rf: int = 2):
        """
        Initialize the hybrid anomaly detector.
        
        Args:
            contamination (float): Expected proportion of anomalies in the data
            n_estimators_iso_forest (int): Number of estimators for Isolation Forest.
            max_features_iso_forest (float): Max features for Isolation Forest.
            n_estimators_rf (int): Number of estimators for Random Forest.
            max_depth_rf (Optional[int]): Max depth for Random Forest.
            min_samples_split_rf (int): Min samples split for Random Forest.
        """
        self.logger = get_logger('model')
        self.logger.info(f"Initializing HybridAnomalyDetector with contamination={contamination}")
        
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=n_estimators_iso_forest,
            max_features=max_features_iso_forest
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=n_estimators_rf,
            max_depth=max_depth_rf,
            min_samples_split=min_samples_split_rf,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.numeric_columns = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, feature_columns: List[str], progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """Train the hybrid model."""
        self.feature_columns = feature_columns
        
        # Filter out datetime columns for scaling and modeling
        numeric_columns = []
        for col in feature_columns:
            if col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]) or pd.api.types.is_object_dtype(X[col]):
                    # Skip datetime and object columns
                    continue
                numeric_columns.append(col)
        
        if not numeric_columns:
            raise ValueError("No numeric columns found for training. Please ensure feature_columns contains numeric data.")
        
        # Store numeric columns for prediction
        self.numeric_columns = numeric_columns
        
        # Use only numeric columns for scaling and training
        X_numeric = X[numeric_columns]
        X_scaled = self.scaler.fit_transform(X_numeric)

        # Train Isolation Forest
        if progress_callback: progress_callback(45, "Fitting Isolation Forest model...")
        self.isolation_forest.fit(X_scaled)
        if progress_callback: progress_callback(50, "Isolation Forest trained.")
        
        # Train Random Forest. Labels for RF are derived from Isolation Forest initial predictions
        if progress_callback: progress_callback(81, "Fitting Random Forest...")
        if_scores_train = -self.isolation_forest.score_samples(X_scaled)
        threshold_for_rf_labels = np.percentile(if_scores_train, (1 - self.contamination) * 100)
        rf_labels = (if_scores_train > threshold_for_rf_labels).astype(int)
        self.random_forest.fit(X_scaled, rf_labels)
        if progress_callback: progress_callback(85, "Random Forest trained.")
        
        self.is_fitted = True
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using both models."""
        if self.feature_columns is None or self.numeric_columns is None:
            raise ValueError("Model must be trained before prediction")
            
        X_scaled = self.scaler.transform(X[self.numeric_columns])
        
        # Isolation Forest predictions
        if_scores = -self.isolation_forest.score_samples(X_scaled)
        if_predictions = (if_scores > np.percentile(if_scores, (1 - self.contamination) * 100)).astype(int)
        
        # Random Forest predictions
        rf_predictions = self.random_forest.predict(X_scaled)
        
        # Combined predictions: Anomaly if at least 2 models predict anomaly (majority vote)
        combined_predictions = (if_predictions + rf_predictions >= 1).astype(int)

        return combined_predictions, if_scores
    
    def explain_anomaly(self, X_anomaly: pd.DataFrame) -> Dict[str, float]:
        """Explains an anomaly by showing the most important features from Random Forest."""
        if self.random_forest is None or self.numeric_columns is None:
            return {"error": "Model not trained or feature columns not set."}

        # Scale the anomalous instance using the *trained* scaler
        X_scaled_anomaly = self.scaler.transform(X_anomaly[self.numeric_columns])

        # Get feature importances from Random Forest
        feature_importances = self.random_forest.feature_importances_
        
        # Create a mapping of feature names to their importances
        importance_dict = dict(zip(self.numeric_columns, feature_importances))
        
        # Sort features by importance in descending order
        sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        
        return {feature: importance for feature, importance in sorted_importance}

    def save(self, path: str) -> None:
        """Save the model components."""
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'random_forest': self.random_forest,
            'feature_columns': self.feature_columns,
            'numeric_columns': self.numeric_columns,
            'contamination': self.contamination,
            'n_estimators_iso_forest': self.isolation_forest.n_estimators,
            'max_features_iso_forest': self.isolation_forest.max_features,
            'n_estimators_rf': self.random_forest.n_estimators,
            'max_depth_rf': self.random_forest.max_depth,
            'min_samples_split_rf': self.random_forest.min_samples_split
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'HybridAnomalyDetector':
        """Load a saved model."""
        model_data = joblib.load(path)
        detector = cls(
            contamination=model_data.get('contamination', 0.1),
            n_estimators_iso_forest=model_data.get('n_estimators_iso_forest', 100),
            max_features_iso_forest=model_data.get('max_features_iso_forest', 1.0),
            n_estimators_rf=model_data.get('n_estimators_rf', 100),
            max_depth_rf=model_data.get('max_depth_rf', 10),
            min_samples_split_rf=model_data.get('min_samples_split_rf', 2)
        )
        detector.scaler = model_data['scaler']
        detector.isolation_forest = model_data['isolation_forest']
        detector.random_forest = model_data['random_forest']
        detector.feature_columns = model_data['feature_columns']
        detector.numeric_columns = model_data['numeric_columns']
        
        return detector

if __name__ == "__main__":
    # Example usage
    from data_generator import IAMLogGenerator
    from feature_engineering import IAMFeatureEngineer
    
    # Generate and prepare data
    generator = IAMLogGenerator()
    df = generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
    
    # Extract features
    engineer = IAMFeatureEngineer()
    df_features = engineer.extract_features(df)
    
    # Train model
    detector = HybridAnomalyDetector()
    detector.fit(df_features, df_features.columns)
    
    # Make predictions
    predictions, scores = detector.predict(df_features)
    
    print("Anomaly Detection Results:")
    print(f"Number of anomalies detected: {predictions.sum()}")
    print(f"True anomalies in data: {df['is_anomaly'].sum()}")
    
    # Calculate accuracy
    accuracy = (predictions == df['is_anomaly']).mean()
    print(f"Accuracy: {accuracy:.2f}") 