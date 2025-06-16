import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from utils.logging_config import get_logger

class SimpleAnomalyDetector:
    def __init__(self, threshold: float = 2.0):
        """
        Initialize a simple anomaly detector based on statistical methods.
        
        Args:
            threshold (float): Number of standard deviations to consider as anomaly
        """
        self.logger = get_logger('model')
        self.logger.info(f"Initializing SimpleAnomalyDetector with threshold={threshold}")
        
        self.threshold = threshold
        self.feature_stats = {}
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Fit the detector by calculating statistics for each feature.
        
        Args:
            X (pd.DataFrame): Training data
        """
        self.logger.info("Fitting simple anomaly detector")
        try:
            # Calculate mean and standard deviation for each feature
            for column in X.columns:
                self.feature_stats[column] = {
                    'mean': X[column].mean(),
                    'std': X[column].std()
                }
            
            self.logger.debug(f"Calculated statistics for {len(self.feature_stats)} features")
            self.is_fitted = True
            self.logger.info("Model fitting completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model fitting: {e}", exc_info=True)
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies in the data.
        
        Args:
            X (pd.DataFrame): Data to predict anomalies for
            
        Returns:
            np.ndarray: Array of predictions (1 for normal, -1 for anomalies)
        """
        self.logger.info("Making predictions")
        if not self.is_fitted:
            self.logger.error("Model not fitted")
            raise ValueError("Model must be fitted before making predictions")
            
        try:
            # Calculate z-scores for each feature
            z_scores = pd.DataFrame(index=X.index)
            for column in X.columns:
                stats = self.feature_stats[column]
                z_scores[column] = (X[column] - stats['mean']) / stats['std']
            
            # Calculate maximum absolute z-score for each row
            max_z_scores = z_scores.abs().max(axis=1)
            
            # Predict anomalies based on threshold
            predictions = np.where(max_z_scores > self.threshold, -1, 1)
            
            self.logger.debug(f"Predictions made: {np.unique(predictions, return_counts=True)}")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}", exc_info=True)
            raise
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores for the data.
        
        Args:
            X (pd.DataFrame): Data to calculate anomaly scores for
            
        Returns:
            np.ndarray: Array of anomaly scores (higher values indicate more anomalous)
        """
        self.logger.info("Calculating anomaly scores")
        if not self.is_fitted:
            self.logger.error("Model not fitted")
            raise ValueError("Model must be fitted before calculating anomaly scores")
            
        try:
            # Calculate z-scores for each feature
            z_scores = pd.DataFrame(index=X.index)
            for column in X.columns:
                stats = self.feature_stats[column]
                z_scores[column] = (X[column] - stats['mean']) / stats['std']
            
            # Calculate maximum absolute z-score for each row
            anomaly_scores = z_scores.abs().max(axis=1).values
            
            self.logger.debug(f"Anomaly scores calculated: min={anomaly_scores.min():.3f}, max={anomaly_scores.max():.3f}, mean={anomaly_scores.mean():.3f}")
            return anomaly_scores
            
        except Exception as e:
            self.logger.error(f"Error during anomaly score calculation: {e}", exc_info=True)
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on standard deviations.
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to importance scores
        """
        self.logger.info("Calculating feature importance")
        if not self.is_fitted:
            self.logger.error("Model not fitted")
            raise ValueError("Model must be fitted before getting feature importance")
            
        try:
            # Use standard deviations as importance scores
            feature_importance = {
                feature: stats['std']
                for feature, stats in self.feature_stats.items()
            }
            
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            self.logger.debug(f"Feature importance calculated: {feature_importance}")
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}", exc_info=True)
            raise 