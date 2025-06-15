import unittest
import pandas as pd
import numpy as np
from src.models.hybrid_model import HybridAnomalyDetector
from src.data_generator import IAMLogGenerator
from src.feature_engineering import FeatureEngineer

class TestHybridAnomalyDetector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.detector = HybridAnomalyDetector()
        self.generator = IAMLogGenerator()
        self.engineer = FeatureEngineer()
        
    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.isolation_forest)
        self.assertIsNotNone(self.detector.random_forest)
        
    def test_model_training(self):
        """Test model training functionality."""
        # Generate and prepare data
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        self.detector.fit(df_features[feature_columns], feature_columns)
        
        # Check that models are trained
        self.assertTrue(hasattr(self.detector.isolation_forest, 'predict'))
        self.assertTrue(hasattr(self.detector.random_forest, 'predict'))
        
    def test_prediction_functionality(self):
        """Test model prediction functionality."""
        # Generate and prepare data
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        self.detector.fit(df_features[feature_columns], feature_columns)
        
        # Make predictions
        predictions, scores = self.detector.predict(df_features)
        
        # Check prediction outputs
        self.assertIsInstance(predictions, np.ndarray)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(predictions), len(df_features))
        self.assertEqual(len(scores), len(df_features))
        
        # Check prediction values
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
        self.assertTrue(all(0 <= score <= 1 for score in scores))
        
    def test_anomaly_detection_accuracy(self):
        """Test anomaly detection accuracy on synthetic data."""
        # Generate data with known anomalies
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        self.detector.fit(df_features[feature_columns], feature_columns)
        
        # Make predictions
        predictions, _ = self.detector.predict(df_features)
        
        # Calculate accuracy metrics
        true_anomalies = df_features['is_anomaly'].values
        accuracy = (predictions == true_anomalies).mean()
        
        # Check that accuracy is reasonable (better than random)
        self.assertGreater(accuracy, 0.5)
        
    def test_model_consistency(self):
        """Test that model predictions are consistent for same input."""
        # Generate and prepare data
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        self.detector.fit(df_features[feature_columns], feature_columns)
        
        # Make predictions twice
        predictions1, scores1 = self.detector.predict(df_features)
        predictions2, scores2 = self.detector.predict(df_features)
        
        # Check consistency
        np.testing.assert_array_equal(predictions1, predictions2)
        np.testing.assert_array_equal(scores1, scores2)
        
    def test_anomaly_explanation(self):
        """Test anomaly explanation functionality."""
        # Generate and prepare data
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        self.detector.fit(df_features[feature_columns], feature_columns)
        
        # Find an anomalous sample
        predictions, _ = self.detector.predict(df_features)
        anomalous_indices = np.where(predictions == 1)[0]
        
        if len(anomalous_indices) > 0:
            # Get explanation for first anomaly
            anomaly_data = df_features.iloc[[anomalous_indices[0]]]
            explanation = self.detector.explain_anomaly(anomaly_data)
            
            # Check explanation format
            self.assertIsInstance(explanation, dict)
            self.assertGreater(len(explanation), 0)
            
            # Check explanation values
            for feature, importance in explanation.items():
                self.assertIsInstance(feature, str)
                self.assertIsInstance(importance, float)
                self.assertTrue(0 <= importance <= 1)
                
    def test_model_parameters(self):
        """Test model parameter handling."""
        # Test with custom parameters
        custom_detector = HybridAnomalyDetector(
            contamination=0.2,
            n_estimators_iso_forest=100,
            max_features_iso_forest=0.5,
            n_estimators_rf=50,
            max_depth_rf=10,
            min_samples_split_rf=5
        )
        
        # Generate and prepare data
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.2)
        df_features = self.engineer.engineer_features(df)
        feature_columns = self.engineer.get_feature_columns()
        
        # Train model
        custom_detector.fit(df_features[feature_columns], feature_columns)
        
        # Make predictions
        predictions, _ = custom_detector.predict(df_features)
        
        # Check that anomaly ratio is close to contamination
        actual_ratio = predictions.mean()
        self.assertAlmostEqual(actual_ratio, 0.2, delta=0.1)
        
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.detector.fit(empty_df, [])
            
        # Test with missing features
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        with self.assertRaises(ValueError):
            self.detector.fit(df_features, ['non_existent_feature'])
            
        # Test with invalid data types
        invalid_df = pd.DataFrame({'invalid': ['not', 'numeric', 'data']})
        with self.assertRaises(ValueError):
            self.detector.fit(invalid_df, ['invalid'])

if __name__ == '__main__':
    unittest.main() 