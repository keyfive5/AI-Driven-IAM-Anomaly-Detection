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
        
        # Add cyclical time features for hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
        
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
                                     'time_since_last_access', 'hour_sin', 'hour_cos'])
        
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

        # Top N IP addresses as categorical features
        top_n_ips = ip_counts.head(20).index.tolist() # Consider top 20 IPs
        df['top_ip'] = df['ip_address'].apply(lambda x: x if x in top_n_ips else 'other_ip')
        
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
        self.categorical_columns.append('top_ip') # Add new categorical feature
        
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

        # Add rate-based features using a robust merge strategy
        df_temp = df[['user_id', 'timestamp', 'action', 'resource', '_row_id']].copy()
        df_temp = df_temp.sort_values(by=['user_id', 'timestamp'])

        # Actions per minute
        actions_per_minute_result = df_temp.groupby('user_id').rolling(
            '1min', on='timestamp', closed='left', min_periods=1
        )['timestamp'].count().rename('actions_per_minute')
        actions_per_minute_df = actions_per_minute_result.reset_index()
        actions_per_minute_df.rename(columns={'level_1': '_row_id'}, inplace=True) # Rename the index level that became a column
        df = df.merge(actions_per_minute_df[['_row_id', 'actions_per_minute']], on='_row_id', how='left')
        df['actions_per_minute'] = df['actions_per_minute'].fillna(0)

        # Unique actions per hour
        unique_actions_per_hour_result = df_temp.groupby('user_id').rolling(
            '1h', on='timestamp', closed='left', min_periods=1
        )['action'].apply(lambda x: x.nunique() if x.nunique() else np.nan).rename('unique_actions_per_hour')
        unique_actions_per_hour_df = unique_actions_per_hour_result.reset_index()
        unique_actions_per_hour_df.rename(columns={'level_1': '_row_id'}, inplace=True)
        df = df.merge(unique_actions_per_hour_df[['_row_id', 'unique_actions_per_hour']], on='_row_id', how='left')
        df['unique_actions_per_hour'] = df['unique_actions_per_hour'].fillna(0)

        # Unique resources per hour
        unique_resources_per_hour_result = df_temp.groupby('user_id').rolling(
            '1h', on='timestamp', closed='left', min_periods=1
        )['resource'].apply(lambda x: x.nunique() if x.nunique() else np.nan).rename('unique_resources_per_hour')
        unique_resources_per_hour_df = unique_resources_per_hour_result.reset_index()
        unique_resources_per_hour_df.rename(columns={'level_1': '_row_id'}, inplace=True)
        df = df.merge(unique_resources_per_hour_df[['_row_id', 'unique_resources_per_hour']], on='_row_id', how='left')
        df['unique_resources_per_hour'] = df['unique_resources_per_hour'].fillna(0)
        
        # Add to feature columns
        self.numerical_columns.extend(action_counts.columns.tolist())
        self.numerical_columns.extend(resource_counts.columns.tolist())
        self.numerical_columns.extend([
            'success_count', 'failure_count', 'success_ratio',
            'actions_per_minute', 'unique_actions_per_hour', 'unique_resources_per_hour' # New features
        ])
        
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
        
        # Average duration between actions in a session (if applicable)
        # Requires sorted data by session_id and timestamp
        # Use _row_id for robust merging
        df_temp_session = df[['session_id', 'timestamp', 'action', '_row_id']].copy()
        df_temp_session = df_temp_session.sort_values(by=['session_id', 'timestamp'])

        time_between_actions_result = df_temp_session.groupby('session_id')['timestamp'].diff().dt.total_seconds().rename('time_between_actions')
        time_between_actions_df = time_between_actions_result.reset_index()
        time_between_actions_df.rename(columns={'level_1': '_row_id'}, inplace=True)
        df = df.merge(time_between_actions_df[['_row_id', 'time_between_actions']], on='_row_id', how='left')
        df['time_between_actions'] = df['time_between_actions'].fillna(0) # Fill NaNs for first action in session

        # Average action duration in session
        # This needs to be re-calculated after time_between_actions is merged to df
        average_action_duration_in_session_result = df.groupby('session_id')['time_between_actions'].transform('mean').rename('average_action_duration_in_session')
        average_action_duration_in_session_df = average_action_duration_in_session_result.reset_index()
        average_action_duration_in_session_df.rename(columns={'level_1': '_row_id'}, inplace=True)
        df = df.merge(average_action_duration_in_session_df[['_row_id', 'average_action_duration_in_session']], on='_row_id', how='left')
        df['average_action_duration_in_session'] = df['average_action_duration_in_session'].fillna(0)

        # Distinct actions count per session
        distinct_actions_per_session_result = df_temp_session.groupby('session_id')['action'].transform('nunique').rename('distinct_actions_per_session')
        distinct_actions_per_session_df = distinct_actions_per_session_result.reset_index()
        distinct_actions_per_session_df.rename(columns={'level_1': '_row_id'}, inplace=True)
        df = df.merge(distinct_actions_per_session_df[['_row_id', 'distinct_actions_per_session']], on='_row_id', how='left')
        df['distinct_actions_per_session'] = df['distinct_actions_per_session'].fillna(0)
        
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
            'avg_session_duration',
            'average_action_duration_in_session', # New feature
            'distinct_actions_per_session' # New feature
        ])
        
        return df
    
    def extract_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on regions."""
        # Region frequency
        region_counts = df['region'].value_counts()
        df['region_frequency'] = df['region'].map(region_counts)

        # Top N regions as categorical features
        top_n_regions = region_counts.head(10).index.tolist() # Consider top 10 regions
        df['top_region'] = df['region'].apply(lambda x: x if x in top_n_regions else 'other_region')
        
        # Region changes per session - make conditional for real logs
        if 'session_id' in df.columns and df['session_id'].count() > 0:
            df['region_changes_in_session'] = df.groupby('session_id')['region'].transform('nunique')
            df['region_changes_in_session'] = df['region_changes_in_session'].fillna(0)
            self.numerical_columns.append('region_changes_in_session')
        else:
            df['region_changes_in_session'] = 0 # Placeholder
        
        # Add to feature columns (only base ones here, region_changes_in_session added conditionally)
        self.numerical_columns.extend(['region_frequency'])
        self.categorical_columns.append('top_region') # Add new categorical feature
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agents."""
        # User agent frequency
        ua_counts = df['user_agent'].value_counts()
        df['user_agent'].fillna('unknown', inplace=True)
        df['user_agent_frequency'] = df['user_agent'].map(ua_counts)

        # Top N User Agents as categorical features
        top_n_uas = ua_counts.head(20).index.tolist() # Consider top 20 user agents
        df['top_user_agent'] = df['user_agent'].apply(lambda x: x if x in top_n_uas else 'other_user_agent')
        
        # User agent changes per session - make conditional for real logs
        if 'session_id' in df.columns and df['session_id'].count() > 0:
            df['user_agent_changes_in_session'] = df.groupby('session_id')['user_agent'].transform('nunique')
            df['user_agent_changes_in_session'] = df['user_agent_changes_in_session'].fillna(0)
            self.numerical_columns.append('user_agent_changes_in_session')
        else:
            df['user_agent_changes_in_session'] = 0 # Placeholder
        
        # Add to feature columns (only base ones here, user_agent_changes_in_session added conditionally)
        self.numerical_columns.extend(['user_agent_frequency'])
        self.categorical_columns.append('top_user_agent') # Add new categorical feature
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        # Reset numerical columns before each run to avoid duplicates if called multiple times
        self.numerical_columns = []
        self.categorical_columns = [] # Reset categorical columns too

        # Ensure DataFrame has a unique index to prevent reindexing errors
        # Add a unique row identifier to merge results robustly
        df['_row_id'] = df.index # Preserve original index as a column
        df = df.reset_index(drop=True)

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
            
            print("Extracting IP-based features...")
            df = self.extract_ip_features(df)
            print(f"DEBUG: Columns after IP features: {df.columns.tolist()}")

            print("Extracting behavioral features...")
            df = self.extract_behavioral_features(df)
            print(f"DEBUG: Columns after behavioral features: {df.columns.tolist()}")
            print(f"DEBUG: Duplicates in ['user_id', 'timestamp'] after behavioral features: {df.duplicated(subset=['user_id', 'timestamp']).any()}")

            print("Extracting session-based features...")
            df = self.extract_session_features(df)
            print(f"DEBUG: Columns after session features: {df.columns.tolist()}")
            print(f"DEBUG: Duplicates in ['user_id', 'timestamp'] after session features: {df.duplicated(subset=['user_id', 'timestamp']).any()}")

            print("Extracting region-based features...")
            df = self.extract_region_features(df)
            print(f"DEBUG: Columns after region features: {df.columns.tolist()}")

            print("Extracting user agent features...")
            df = self.extract_user_agent_features(df)
            print(f"DEBUG: Columns after user agent features: {df.columns.tolist()}")
            
            # Convert categorical features to numerical using one-hot encoding
            if self.categorical_columns:
                print(f"Applying one-hot encoding for categorical columns: {self.categorical_columns}")
                df = pd.get_dummies(df, columns=self.categorical_columns, dummy_na=False) # Use dummy_na=False to avoid NaN columns
                
                # Add newly created dummy columns to numerical_columns
                # Filter for columns that actually exist in df (get_dummies might not create all if categories are missing)
                new_dummy_columns = [col for col in df.columns if any(cat_col + '_' in col for cat_col in self.categorical_columns)]
                self.numerical_columns.extend(new_dummy_columns)
                print(f"DEBUG: Columns after one-hot encoding: {df.columns.tolist()}")

            # Final cleanup: fill any remaining NaNs in numerical columns
            for col in self.numerical_columns:
                if col not in df.columns:
                    print(f"Warning: Numerical feature '{col}' not found in DataFrame after feature engineering. It might have been dropped or not created. Skipping NaN fill for this column.")
                    continue
                if df[col].isnull().any():
                    print(f"Filling NaN values in numerical column: {col}")
                    df[col] = df[col].fillna(0) # Or a more sophisticated imputation strategy

            # Remove original datetime and identifier columns which are not features
            cols_to_drop = ['timestamp', 'session_start', 'session_end', 'session_id',
                            'user_id', 'action', 'resource', 'ip_address', 'region', 'user_agent', 'status']
            # Filter out columns that don't exist in the DataFrame
            cols_to_drop = [col for col in cols_to_drop if col in df.columns]
            df = df.drop(columns=cols_to_drop, errors='ignore')

            # Filter self.numerical_columns to only include columns actually present in the final DataFrame
            self.numerical_columns = [col for col in self.numerical_columns if col in df.columns]

            # Drop the temporary row_id column
            df = df.drop(columns=['_row_id'], errors='ignore')

            print(f"DEBUG: Final numerical columns to be used by model: {self.numerical_columns}")

            return df

        except Exception as e:
            print(f"An error occurred during feature engineering: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            raise # Re-raise the exception after printing

    def get_feature_columns(self) -> List[str]:
        """Return the list of engineered feature columns."""
        return self.numerical_columns

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
    print(f"Total number of features: {len(engineer.numerical_columns)}")
    print("\nFeature columns:")
    print(engineer.numerical_columns)
    
    print("\nSample of engineered data:")
    print(df_with_features[engineer.numerical_columns].head()) 