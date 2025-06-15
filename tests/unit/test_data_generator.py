import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_generator import IAMLogGenerator

class TestIAMLogGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = IAMLogGenerator()
        
    def test_generate_dataset_basic(self):
        """Test basic dataset generation with default parameters."""
        n_events = 1000
        anomaly_ratio = 0.1
        
        df = self.generator.generate_dataset(n_events=n_events, anomaly_ratio=anomaly_ratio)
        
        # Check DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Check required columns
        required_columns = ['timestamp', 'user_id', 'action', 'resource', 'ip_address', 
                          'region', 'status', 'session_id', 'session_start', 'session_end', 
                          'user_agent', 'is_anomaly']
        for col in required_columns:
            self.assertIn(col, df.columns)
            
        # Check anomaly ratio
        actual_anomaly_ratio = df['is_anomaly'].mean()
        self.assertAlmostEqual(actual_anomaly_ratio, anomaly_ratio, delta=0.05)
        
    def test_generate_dataset_custom_size(self):
        """Test dataset generation with custom number of events."""
        n_events = 500
        df = self.generator.generate_dataset(n_events=n_events)
        self.assertEqual(len(df), n_events)
        
    def test_timestamp_consistency(self):
        """Test that timestamps are consistent within sessions."""
        df = self.generator.generate_dataset(n_events=100)
        
        # Group by session_id and check timestamp ordering
        for session_id, group in df.groupby('session_id'):
            timestamps = pd.to_datetime(group['timestamp'])
            session_start = pd.to_datetime(group['session_start'].iloc[0])
            session_end = pd.to_datetime(group['session_end'].iloc[0])
            
            # Check all timestamps are within session bounds
            self.assertTrue(all(session_start <= ts <= session_end for ts in timestamps))
            
            # Check timestamps are in ascending order
            self.assertTrue(all(timestamps.iloc[i] <= timestamps.iloc[i+1] 
                              for i in range(len(timestamps)-1)))
            
    def test_ip_address_patterns(self):
        """Test IP address patterns for normal and anomalous events."""
        df = self.generator.generate_dataset(n_events=1000, anomaly_ratio=0.3)
        
        # Check IP patterns
        normal_ips = df[~df['is_anomaly']]['ip_address']
        anomalous_ips = df[df['is_anomaly']]['ip_address']
        
        # Normal IPs should be in 192.168.x.x range
        self.assertTrue(all(ip.startswith('192.168.') for ip in normal_ips))
        
        # Anomalous IPs should be in 203.0.x.x range
        self.assertTrue(all(ip.startswith('203.0.') for ip in anomalous_ips))
        
    def test_user_role_consistency(self):
        """Test that users are assigned consistent roles."""
        df = self.generator.generate_dataset(n_events=1000)
        
        # Group by user_id and check role consistency
        for user_id, group in df.groupby('user_id'):
            roles = group['role'].unique()
            self.assertEqual(len(roles), 1, f"User {user_id} has multiple roles: {roles}")
            
    def test_action_permissions(self):
        """Test that actions are consistent with role permissions."""
        df = self.generator.generate_dataset(n_events=1000)
        
        # For each user, check their actions against their role permissions
        for user_id, group in df.groupby('user_id'):
            role = group['role'].iloc[0]
            actions = group['action'].unique()
            
            # Get role permissions
            role_permissions = self.generator.role_permissions[role]
            
            # For normal events, all actions should be in role permissions
            normal_actions = group[~group['is_anomaly']]['action'].unique()
            self.assertTrue(all(action in role_permissions for action in normal_actions))
            
            # For anomalous events, at least some actions should be outside role permissions
            anomalous_actions = group[group['is_anomaly']]['action'].unique()
            self.assertTrue(any(action not in role_permissions for action in anomalous_actions))
            
    def test_data_types(self):
        """Test that all columns have correct data types."""
        df = self.generator.generate_dataset(n_events=100)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['session_start']))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['session_end']))
        self.assertTrue(pd.api.types.is_bool_dtype(df['is_anomaly']))
        
    def test_missing_values(self):
        """Test that there are no unexpected missing values."""
        df = self.generator.generate_dataset(n_events=100)
        
        # Check for missing values in required columns
        required_columns = ['timestamp', 'user_id', 'action', 'resource', 'ip_address', 
                          'region', 'status', 'session_id', 'session_start', 'session_end', 
                          'user_agent', 'is_anomaly']
        for col in required_columns:
            self.assertEqual(df[col].isnull().sum(), 0, f"Column {col} has missing values")

if __name__ == '__main__':
    unittest.main() 