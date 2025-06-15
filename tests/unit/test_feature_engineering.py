import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import FeatureEngineer
from src.data_generator import IAMLogGenerator

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.engineer = FeatureEngineer()
        self.generator = IAMLogGenerator()
        
    def test_basic_feature_engineering(self):
        """Test basic feature engineering functionality."""
        # Generate sample data
        df = self.generator.generate_dataset(n_events=1000)
        
        # Engineer features
        df_features = self.engineer.engineer_features(df)
        
        # Check DataFrame structure
        self.assertIsInstance(df_features, pd.DataFrame)
        self.assertGreater(len(df_features), 0)
        
        # Check that original columns are preserved
        original_columns = df.columns
        for col in original_columns:
            self.assertIn(col, df_features.columns)
            
        # Check that new features are added
        self.assertGreater(len(df_features.columns), len(original_columns))
        
    def test_time_based_features(self):
        """Test generation of time-based features."""
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        
        # Check time-based features
        time_features = [col for col in df_features.columns if 'time' in col.lower()]
        self.assertGreater(len(time_features), 0)
        
        # Check that time features are numeric
        for feature in time_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_features[feature]))
            
    def test_session_features(self):
        """Test generation of session-based features."""
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        
        # Check session-based features
        session_features = [col for col in df_features.columns if 'session' in col.lower()]
        self.assertGreater(len(session_features), 0)
        
        # Check that session features are numeric
        for feature in session_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_features[feature]))
            
    def test_user_based_features(self):
        """Test generation of user-based features."""
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        
        # Check user-based features
        user_features = [col for col in df_features.columns if 'user' in col.lower()]
        self.assertGreater(len(user_features), 0)
        
        # Check that user features are numeric
        for feature in user_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_features[feature]))
            
    def test_action_based_features(self):
        """Test generation of action-based features."""
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        
        # Check action-based features
        action_features = [col for col in df_features.columns if 'action' in col.lower()]
        self.assertGreater(len(action_features), 0)
        
        # Check that action features are numeric
        for feature in action_features:
            self.assertTrue(pd.api.types.is_numeric_dtype(df_features[feature]))
            
    def test_missing_value_handling(self):
        """Test handling of missing values in feature engineering."""
        # Generate data with some missing values
        df = self.generator.generate_dataset(n_events=100)
        df.loc[0:10, 'ip_address'] = np.nan
        df.loc[20:30, 'user_agent'] = np.nan
        
        # Engineer features
        df_features = self.engineer.engineer_features(df)
        
        # Check that no NaN values exist in engineered features
        engineered_columns = [col for col in df_features.columns 
                            if col not in df.columns]
        for col in engineered_columns:
            self.assertEqual(df_features[col].isnull().sum(), 0,
                           f"Column {col} has missing values")
            
    def test_feature_scaling(self):
        """Test that features are properly scaled."""
        df = self.generator.generate_dataset(n_events=100)
        df_features = self.engineer.engineer_features(df)
        
        # Get numerical columns
        numerical_columns = df_features.select_dtypes(include=[np.number]).columns
        
        # Check that numerical features are scaled
        for col in numerical_columns:
            if col not in ['is_anomaly']:  # Skip target variable
                # Check that values are not too extreme
                self.assertLess(df_features[col].max(), 1000)
                self.assertGreater(df_features[col].min(), -1000)
                
    def test_feature_consistency(self):
        """Test that features are consistent across multiple runs."""
        df = self.generator.generate_dataset(n_events=100)
        
        # Generate features twice
        df_features1 = self.engineer.engineer_features(df)
        df_features2 = self.engineer.engineer_features(df)
        
        # Check that features are identical
        pd.testing.assert_frame_equal(df_features1, df_features2)
        
    def test_get_feature_columns(self):
        """Test the get_feature_columns method."""
        df = self.generator.generate_dataset(n_events=100)
        self.engineer.engineer_features(df)
        
        feature_columns = self.engineer.get_feature_columns()
        
        # Check that feature columns are returned
        self.assertIsInstance(feature_columns, list)
        self.assertGreater(len(feature_columns), 0)
        
        # Check that all returned columns exist in the DataFrame
        for col in feature_columns:
            self.assertIn(col, df.columns)

if __name__ == '__main__':
    unittest.main() 