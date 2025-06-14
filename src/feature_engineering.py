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
            # Initialize column if it might be used later, but don't add to numerical_columns if not truly a feature
            df['session_duration'] = 0 
        
        # Add to feature columns (only base ones here, session_duration added conditionally above)
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
        if 'session_id' in df.columns and df['session_id'].count() > 0:
            df['ip_changes_in_session'] = df.groupby('session_id')['ip_address'].transform(
                lambda x: x.nunique()
            )
            df['ip_changes_in_session'] = df['ip_changes_in_session'].fillna(0) # Fill NaN from transform
            self.numerical_columns.append('ip_changes_in_session')
        else:
            df['ip_changes_in_session'] = 0 # Placeholder
        
        # Add to feature columns (only base ones here, ip_changes_in_session added conditionally above)
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
        # Check if session_id or session_duration columns are entirely NaN or missing, if so, skip
        if df['session_id'].count() == 0 or df['session_duration'].count() == 0:
            print("Skipping session-based feature extraction: 'session_id' or 'session_duration' column contains no valid data.")
            # Ensure columns are created as placeholders so later steps don't fail, but not added to numerical_columns
            df['actions_per_session'] = 0
            df['unique_resources_per_session'] = 0
            df['sessions_per_user'] = 0
            df['avg_session_duration'] = 0
            return df # Return original DataFrame if session columns are missing

        # Actions per session
        df['actions_per_session'] = df.groupby('session_id').size()
        df['actions_per_session'] = df['actions_per_session'].fillna(0) # Fill NaNs from merge/groupby
        
        # Unique resources accessed per session
        df['unique_resources_per_session'] = df.groupby('session_id')['resource'].transform('nunique')
        df['unique_resources_per_session'] = df['unique_resources_per_session'].fillna(0)
        
        # Session frequency per user
        df['sessions_per_user'] = df.groupby('user_id')['session_id'].transform('nunique')
        df['sessions_per_user'] = df['sessions_per_user'].fillna(0)
        
        # Average session duration per user
        df['avg_session_duration'] = df.groupby('user_id')['session_duration'].transform('mean')
        df['avg_session_duration'] = df['avg_session_duration'].fillna(0)
        
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
        
        # Region changes per session - make conditional for real logs
        if 'session_id' in df.columns and df['session_id'].count() > 0:
            df['region_changes_in_session'] = df.groupby('session_id')['region'].transform('nunique')
            df['region_changes_in_session'] = df['region_changes_in_session'].fillna(0)
            self.numerical_columns.append('region_changes_in_session')
        else:
            df['region_changes_in_session'] = 0 # Placeholder
        
        # Add to feature columns (only base ones here, region_changes_in_session added conditionally)
        self.numerical_columns.extend(['region_frequency'])
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agents."""
        # User agent frequency
        ua_counts = df['user_agent'].value_counts()
        df['user_agent_frequency'] = df['user_agent'].map(ua_counts)
        
        # User agent changes per session - make conditional for real logs
        if 'session_id' in df.columns and df['session_id'].count() > 0:
            df['user_agent_changes_in_session'] = df.groupby('session_id')['user_agent'].transform('nunique')
            df['user_agent_changes_in_session'] = df['user_agent_changes_in_session'].fillna(0)
            self.numerical_columns.append('user_agent_changes_in_session')
        else:
            df['user_agent_changes_in_session'] = 0 # Placeholder
        
        # Add to feature columns (only base ones here, user_agent_changes_in_session added conditionally)
        self.numerical_columns.extend(['user_agent_frequency'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        # Reset numerical columns before each run to avoid duplicates if called multiple times
        self.numerical_columns = []

        # Ensure essential columns for session-based features exist, fill with NaN if missing
        # This prevents KeyError when df['session_id'] or similar is accessed
        if 'session_id' not in df.columns:
            df['session_id'] = np.nan
        if 'session_start' not in df.columns:
            df['session_start'] = pd.NaT # Use NaT for datetime NaNs
        if 'session_end' not in df.columns:
            df['session_end'] = pd.NaT # Use NaT for datetime NaNs

        print(f"DEBUG: Initial df columns: {df.columns.tolist()}")
        print(f"DEBUG: session_id value_counts (including NaNs) after initial check:\n{df['session_id'].value_counts(dropna=False)}")

        try:
            print("Extracting time-based features...")
            df = self.extract_time_features(df)
            print(f"DEBUG: Columns after time features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_time_features: {e}")
            raise
        
        try:
            print("Extracting IP-based features...")
            df = self.extract_ip_features(df)
            print(f"DEBUG: Columns after IP features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_ip_features: {e}")
            raise
        
        try:
            print("Extracting behavioral features...")
            df = self.extract_behavioral_features(df)
            print(f"DEBUG: Columns after behavioral features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_behavioral_features: {e}")
            raise
        
        try:
            print("Extracting session-based features...")
            df = self.extract_session_features(df)
            print(f"DEBUG: Columns after session features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_session_features: {e}")
            raise
        
        try:
            print("Extracting region-based features...")
            df = self.extract_region_features(df)
            print(f"DEBUG: Columns after region features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_region_features: {e}")
            raise
        
        try:
            print("Extracting user agent features...")
            df = self.extract_user_agent_features(df)
            print(f"DEBUG: Columns after user agent features: {df.columns.tolist()}")
        except Exception as e:
            print(f"ERROR in extract_user_agent_features: {e}")
            raise
        
        # Fill any remaining NaN values with 0
        df = df.fillna(0)
        
        # Get final feature set
        # Ensure only columns that truly exist and are numerical are in feature_columns
        # Filter out any non-numeric or temporary placeholder columns that shouldn't be features
        final_numerical_columns = [col for col in self.numerical_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        self.feature_columns = list(set(final_numerical_columns)) # Use set to remove duplicates, then convert to list

        # Ensure all feature columns have numeric types before passing to models
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) # Coerce to numeric, fill any new NaNs
        
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