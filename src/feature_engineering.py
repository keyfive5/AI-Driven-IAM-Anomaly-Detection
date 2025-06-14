import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import ipaddress
from collections import defaultdict

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamps."""
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_working_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Time since last access for each user
        df = df.sort_values(['user_id', 'timestamp'])
        df['time_since_last_access'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_access'] = df['time_since_last_access'].fillna(0)
        
        # Session-based features - make conditional for real logs
        if 'session_start' in df.columns and 'session_end' in df.columns:
            df['session_duration'] = (df['session_end'] - df['session_start']).dt.total_seconds()
            df['session_duration'] = df['session_duration'].fillna(0)
            self.numerical_columns.append('session_duration')
        else:
            df['session_duration'] = 0 # Placeholder if not available
        
        # Add to feature columns
        self.numerical_columns.extend(['hour', 'day_of_week', 'is_weekend', 'is_working_hour',
                                     'time_since_last_access'])
        
        return df
    
    def extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from IP addresses."""
        def is_private_ip(ip):
            try:
                return ipaddress.ip_address(ip).is_private
            except:
                return False
        
        # IP type features
        df['is_private_ip'] = df['ip_address'].apply(is_private_ip).astype(int)
        
        # IP frequency features
        ip_counts = df['ip_address'].value_counts()
        df['ip_frequency'] = df['ip_address'].map(ip_counts)
        
        # IP changes per session - make conditional for real logs
        if 'session_id' in df.columns:
            df['ip_changes_in_session'] = df.groupby('session_id')['ip_address'].transform(
                lambda x: x.nunique()
            )
            self.numerical_columns.append('ip_changes_in_session')
        else:
            df['ip_changes_in_session'] = 0 # Placeholder if not available
        
        # Add to feature columns
        self.numerical_columns.extend(['is_private_ip', 'ip_frequency'])
        
        return df
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features based on user actions."""
        # Action frequency per user
        action_counts = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
        action_counts.columns = [f'action_{col}_count' for col in action_counts.columns]
        df = df.merge(action_counts, on='user_id', how='left')
        
        # Resource access patterns
        resource_counts = df.groupby(['user_id', 'resource']).size().unstack(fill_value=0)
        resource_counts.columns = [f'resource_{col}_count' for col in resource_counts.columns]
        df = df.merge(resource_counts, on='user_id', how='left')
        
        # Success/failure ratio
        df['success_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'success').sum()
        )
        df['failure_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'failure').sum()
        )
        df['success_ratio'] = df['success_count'] / (df['success_count'] + df['failure_count'])
        df['success_ratio'] = df['success_ratio'].fillna(1)
        
        # Add to feature columns
        self.numerical_columns.extend(action_counts.columns.tolist())
        self.numerical_columns.extend(resource_counts.columns.tolist())
        self.numerical_columns.extend(['success_count', 'failure_count', 'success_ratio'])
        
        return df
    
    def extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract session-based features."""
        if 'session_id' not in df.columns or 'session_duration' not in df.columns:
            print("Skipping session-based feature extraction: 'session_id' or 'session_duration' column missing.")
            return df # Return original DataFrame if session columns are missing

        # Actions per session
        df['actions_per_session'] = df.groupby('session_id').size()
        
        # Unique resources accessed per session
        df['unique_resources_per_session'] = df.groupby('session_id')['resource'].transform('nunique')
        
        # Session frequency per user
        df['sessions_per_user'] = df.groupby('user_id')['session_id'].transform('nunique')
        
        # Average session duration per user
        df['avg_session_duration'] = df.groupby('user_id')['session_duration'].transform('mean')
        
        # Add to feature columns
        self.numerical_columns.extend([
            'actions_per_session',
            'unique_resources_per_session',
            'sessions_per_user',
            'avg_session_duration'
        ])
        
        return df
    
    def extract_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on regions."""
        # Region frequency
        region_counts = df['region'].value_counts()
        df['region_frequency'] = df['region'].map(region_counts)
        
        # Region changes per session
        df['region_changes_in_session'] = df.groupby('session_id')['region'].transform('nunique')
        
        # Add to feature columns
        self.numerical_columns.extend(['region_frequency', 'region_changes_in_session'])
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agents."""
        # User agent frequency
        ua_counts = df['user_agent'].value_counts()
        df['user_agent_frequency'] = df['user_agent'].map(ua_counts)
        
        # User agent changes per session
        df['user_agent_changes_in_session'] = df.groupby('session_id')['user_agent'].transform('nunique')
        
        # Add to feature columns
        self.numerical_columns.extend(['user_agent_frequency', 'user_agent_changes_in_session'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        print("Extracting time-based features...")
        df = self.extract_time_features(df)
        
        print("Extracting IP-based features...")
        df = self.extract_ip_features(df)
        
        print("Extracting behavioral features...")
        df = self.extract_behavioral_features(df)
        
        print("Extracting session-based features...")
        df = self.extract_session_features(df)
        
        print("Extracting region-based features...")
        df = self.extract_region_features(df)
        
        print("Extracting user agent features...")
        df = self.extract_user_agent_features(df)
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        # Get final feature set
        self.feature_columns = self.numerical_columns
        
        return df
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of engineered feature columns."""
        return self.feature_columns

if __name__ == "__main__":
    # Example usage
    from data_generator import IAMLogGenerator
    
    # Generate sample data
    generator = IAMLogGenerator()
    df = generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_with_features = engineer.engineer_features(df)
    
    print("\nEngineered features:")
    print(f"Total number of features: {len(engineer.feature_columns)}")
    print("\nFeature columns:")
    print(engineer.feature_columns)
    
    print("\nSample of engineered data:")
    print(df_with_features[engineer.feature_columns].head()) 