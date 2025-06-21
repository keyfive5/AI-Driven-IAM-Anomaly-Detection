import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import ipaddress
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from utils.logging_config import get_logger

class FeatureEngineer:
    def __init__(self):
        self.logger = get_logger('analysis')
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.scaler = StandardScaler()
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamps."""
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Basic time features
        df_temp = df.dropna(subset=['timestamp']).copy()

        df['hour'] = df_temp['timestamp'].dt.hour
        df['day_of_week'] = df_temp['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_working_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Add cyclical time features for hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'].fillna(0)/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'].fillna(0)/24)
        
        # Time since last access for each user - ensure sorting includes _temp_row_id if present
        sort_cols = ['user_id', 'timestamp']
        if '_temp_row_id' in df.columns:
            sort_cols.append('_temp_row_id')
        df = df.sort_values(sort_cols)
        # Ensure user_id is not NA for groupby. Fill with a placeholder string if needed
        df['user_id_filled'] = df['user_id'].fillna('unknown_user')
        df['time_since_last_access'] = df.groupby('user_id_filled')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_access'] = df['time_since_last_access'].fillna(0)
        df.drop(columns=['user_id_filled'], inplace=True)

        # Session-based features - 'session_start' and 'session_end' are now guaranteed to exist
        df['session_duration'] = (df['session_end'].fillna(df['timestamp']) - df['session_start'].fillna(df['timestamp'])).dt.total_seconds()
        df['session_duration'] = df['session_duration'].fillna(0)
        
        self.numerical_columns.extend([
            'hour', 'day_of_week', 'is_weekend', 'is_working_hour',
            'time_since_last_access', 'hour_sin', 'hour_cos', 'session_duration'
        ])
        
        return df
    
    def extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from IP addresses."""
        def is_private_ip(ip):
            try:
                if isinstance(ip, str) and ip not in ('unknown', ''):
                    return ipaddress.ip_address(ip).is_private
                return False
            except ValueError:
                return False
            except Exception as e:
                return False
        
        df['ip_address_str'] = df['ip_address'].astype(str).fillna('unknown')
        df['is_private_ip'] = df['ip_address_str'].apply(is_private_ip).astype(int)
        ip_counts = df['ip_address_str'].value_counts()
        df['ip_frequency'] = df['ip_address_str'].map(ip_counts).fillna(0)
        top_n_ips = ip_counts.head(20).index.tolist()
        df['top_ip'] = df['ip_address_str'].apply(lambda x: x if x in top_n_ips else 'other_ip')
        # Patch: Only use columns that exist for dropna/sort
        subset_cols = ['session_id', 'ip_address_str']
        sort_cols = ['session_id', 'timestamp']
        if '_temp_row_id' in df.columns:
            subset_cols.append('_temp_row_id')
            sort_cols.append('_temp_row_id')
        df_temp = df.dropna(subset=subset_cols).sort_values(by=sort_cols).copy()
        df_temp['session_id_filled'] = df_temp['session_id'].fillna('unknown_session')
        ip_changes_in_session_rolled = df_temp.groupby('session_id_filled')['ip_address_str'].transform('nunique')
        if '_temp_row_id' in df_temp.columns:
            df_temp = df_temp.set_index('_temp_row_id')
            df_temp['ip_changes_in_session'] = ip_changes_in_session_rolled
            df_temp['ip_changes_in_session'] = df_temp['ip_changes_in_session'].fillna(0)
            df = df.merge(df_temp.reset_index()[['_temp_row_id', 'ip_changes_in_session']], on='_temp_row_id', how='left').fillna({'ip_changes_in_session': 0})
        else:
            df_temp['ip_changes_in_session'] = ip_changes_in_session_rolled
            df_temp['ip_changes_in_session'] = df_temp['ip_changes_in_session'].fillna(0)
            df = df.merge(df_temp[['session_id', 'ip_changes_in_session']], on='session_id', how='left').fillna({'ip_changes_in_session': 0})
        self.numerical_columns.extend(['is_private_ip', 'ip_frequency', 'ip_changes_in_session'])
        self.categorical_columns.append('top_ip')
        df.drop(columns=['ip_address_str'], inplace=True)
        return df
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features based on user actions."""
        self.logger.debug("Extracting behavioral features")
        print(f"DEBUG extract_behavioral_features: Columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_behavioral_features: '_temp_row_id' in df.columns at start: {'_temp_row_id' in df.columns}")
        print(f"DEBUG: extract_behavioral_features - df.index.is_unique at start: {df.index.is_unique}") # DEBUG PRINT

        # Ensure 'user_id', 'action', 'resource', 'status' are strings and filled
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['status'] = df['status'].fillna('unknown_status').astype(str)

        # Action frequency per user
        print(f"DEBUG: Behavioral Features - Before action_counts: Columns={df.columns.tolist()}, Index={df.index.name}")
        action_counts = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
        action_counts.columns = [f'action_{col}_count' for col in action_counts.columns]
        df = df.merge(action_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG: Behavioral Features - After action_counts merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Resource access patterns
        print(f"DEBUG: Behavioral Features - Before resource_counts: Columns={df.columns.tolist()}, Index={df.index.name}")
        resource_counts = df.groupby(['user_id', 'resource']).size().unstack(fill_value=0)
        resource_counts.columns = [f'resource_{col}_count' for col in resource_counts.columns]
        df = df.merge(resource_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG: Behavioral Features - After resource_counts merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Success/failure ratio
        print(f"DEBUG: Behavioral Features - Before success/failure counts: Columns={df.columns.tolist()}, Index={df.index.name}")
        df['success_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'success').sum()
        ).fillna(0)
        df['failure_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'failure').sum()
        ).fillna(0)
        df['success_ratio'] = df['success_count'] / (df['success_count'] + df['failure_count'])
        df['success_ratio'] = df['success_ratio'].fillna(1) # Fill NaN where total count is 0 (divide by zero)
        print(f"DEBUG: Behavioral Features - After success/failure ratio: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Add rate-based features (optimized using rolling windows)
        # Ensure df is sorted by user_id, timestamp, and _temp_row_id for correct rolling window calculation
        print(f"DEBUG: Behavioral Features - Before df_temp_behavioral creation: Columns={df.columns.tolist()}, Index={df.index.name}")
        # Keep _temp_row_id as a column, not index, for rolling operations with 'on' parameter
        df_temp_behavioral = df.sort_values(by=['user_id', 'timestamp', '_temp_row_id']).copy()
        print(f"DEBUG: Behavioral Features - After df_temp_behavioral creation (temp_row_id as column): Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        
        # Convert 'action' and 'resource' to category type for robust rolling window operations
        df_temp_behavioral['action'] = df_temp_behavioral['action'].astype('category')
        df_temp_behavioral['resource'] = df_temp_behavioral['resource'].astype('category')
        print(f"DEBUG: Behavioral Features - dtypes after category conversion: {df_temp_behavioral[['action', 'resource']].dtypes}")

        # Create numerical codes for categorical columns for rolling nunique operations
        df_temp_behavioral['action_code'] = df_temp_behavioral['action'].cat.codes
        df_temp_behavioral['resource_code'] = df_temp_behavioral['resource'].cat.codes
        print(f"DEBUG: Behavioral Features - dtypes after coding: {df_temp_behavioral[['action_code', 'resource_code']].dtypes}")

        # Actions per minute
        print(f"DEBUG: Behavioral Features - Before actions_per_minute rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        print(f"DEBUG: Behavioral Features - dtypes before actions_per_minute rolling: {df_temp_behavioral[['action', 'action_code', 'timestamp', 'user_id', '_temp_row_id']].dtypes}")
        # The groupby should include _temp_row_id to ensure the original index is preserved after reset_index
        actions_per_minute_rolled = df_temp_behavioral.groupby(['user_id', '_temp_row_id']).rolling('1min', on='timestamp')['action'].count().reset_index()
        actions_per_minute_df = actions_per_minute_rolled.rename(columns={'action': 'actions_per_minute'})
        print(f"DEBUG: extract_behavioral_features - df.index.is_unique before actions_per_minute merge: {df.index.is_unique}") # DEBUG PRINT
        print(f"DEBUG: Behavioral Features - actions_per_minute_df columns: {actions_per_minute_df.columns.tolist()}")
        # Ensure merge is on _temp_row_id which is now a column in actions_per_minute_df
        df = df.merge(actions_per_minute_df[['_temp_row_id', 'actions_per_minute']], on='_temp_row_id', how='left').fillna({'actions_per_minute': 0})
        print(f"DEBUG: Behavioral Features - After actions_per_minute merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Unique actions per hour
        print(f"DEBUG: Behavioral Features - Before unique_actions_per_hour rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        print(f"DEBUG: Behavioral Features - dtypes before unique_actions_per_hour rolling: {df_temp_behavioral[['action', 'action_code', 'timestamp', 'user_id', '_temp_row_id']].dtypes}")
        # Use action_code for rolling nunique, ensure pd.Series(x) and raw=False
        unique_actions_per_hour_rolled = df_temp_behavioral.groupby(['user_id', '_temp_row_id']).rolling('1h', on='timestamp')['action_code'].apply(lambda x: pd.Series(x).nunique(), raw=False).reset_index()
        unique_actions_per_hour_df = unique_actions_per_hour_rolled.rename(columns={'action_code': 'unique_actions_per_hour'})
        print(f"DEBUG: Behavioral Features - unique_actions_per_hour_df columns: {unique_actions_per_hour_df.columns.tolist()}")
        # Merge the results back to the original df using _temp_row_id
        df = df.merge(unique_actions_per_hour_df[['_temp_row_id', 'unique_actions_per_hour']], on='_temp_row_id', how='left').fillna({'unique_actions_per_hour': 0})
        print(f"DEBUG: Behavioral Features - After unique_actions_per_hour merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Unique resources per hour
        print(f"DEBUG: Behavioral Features - Before unique_resources_per_hour rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        print(f"DEBUG: Behavioral Features - dtypes before unique_resources_per_hour rolling: {df_temp_behavioral[['resource', 'resource_code', 'timestamp', 'user_id', '_temp_row_id']].dtypes}")
        # Use resource_code for rolling nunique, ensure pd.Series(x) and raw=False
        unique_resources_per_hour_rolled = df_temp_behavioral.groupby(['user_id', '_temp_row_id']).rolling('1h', on='timestamp')['resource_code'].apply(lambda x: pd.Series(x).nunique(), raw=False).reset_index()
        unique_resources_per_hour_df = unique_resources_per_hour_rolled.rename(columns={'resource_code': 'unique_resources_per_hour'})
        print(f"DEBUG: Behavioral Features - unique_resources_per_hour_df columns: {unique_resources_per_hour_df.columns.tolist()}")
        # Merge the results back to the original df using _temp_row_id
        df = df.merge(unique_resources_per_hour_df[['_temp_row_id', 'unique_resources_per_hour']], on='_temp_row_id', how='left').fillna({'unique_resources_per_hour': 0})

        # Drop the temporary code columns after merging
        df_temp_behavioral.drop(columns=['action_code', 'resource_code'], inplace=True)

        # Add to feature columns
        self.numerical_columns.extend(action_counts.columns.tolist())
        self.numerical_columns.extend(resource_counts.columns.tolist())
        self.numerical_columns.extend([
            'success_count', 'failure_count', 'success_ratio',
            'actions_per_minute', 'unique_actions_per_hour', 'unique_resources_per_hour'
        ])

        return df
    
    def extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract session-based features."""
        self.logger.debug("Extracting session features")
        print(f"DEBUG extract_session_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_session_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_session_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure 'session_id', 'resource', 'timestamp', 'action' are filled and correctly typed
        df['session_id'] = df['session_id'].fillna('unknown_session').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        # Ensure timestamp is datetime, already done by IAMLogReader standardization, but fill NaT for calculation robustness
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows where timestamp is NaT as they are crucial for time-based calculations
        # Include _temp_row_id in sorting to maintain its relative order within groups if timestamps are identical
        df_temp = df.dropna(subset=['timestamp', 'session_id', '_temp_row_id']).sort_values(by=['session_id', 'timestamp', '_temp_row_id']).copy()
        print(f"DEBUG extract_session_features: df_temp columns after sort/copy: {df_temp.columns.tolist()}")
        print(f"DEBUG extract_session_features: df_temp index after sort/copy: {df_temp.index.name}")

        # 1. Actions per session
        df_temp['actions_per_session'] = df_temp.groupby('session_id')['session_id'].transform('size')
        print(f"DEBUG extract_session_features: df_temp columns after actions_per_session: {df_temp.columns.tolist()}")

        # 2. Distinct actions per session
        df_temp['distinct_actions_per_session'] = df_temp.groupby('session_id')['action'].transform('nunique')
        print(f"DEBUG extract_session_features: df_temp columns after distinct_actions_per_session: {df_temp.columns.tolist()}")

        # 3. Time between actions in session (for each user within a session)
        df_temp['time_between_actions_in_session'] = df_temp.groupby('session_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        print(f"DEBUG extract_session_features: df_temp columns after time_between_actions: {df_temp.columns.tolist()}")

        # 4. Average time between actions in session (Optimized using rolling mean)
        # Calculate rolling mean directly on df_temp using transform to maintain alignment
        df_temp['average_time_between_actions_in_session'] = df_temp.groupby('session_id')['time_between_actions_in_session'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df_temp['average_time_between_actions_in_session'] = df_temp['average_time_between_actions_in_session'].fillna(0)
        
        # Now, df_temp has all the necessary session features. Merge only relevant columns back to the main df.
        print(f"DEBUG extract_session_features: Before merging session_features_to_merge. df columns: {df.columns.tolist()}")
        # Select only the columns to be merged from df_temp, including _temp_row_id
        session_features_to_merge = df_temp[['_temp_row_id', 'actions_per_session', 'distinct_actions_per_session', 'time_between_actions_in_session', 'average_time_between_actions_in_session']].copy()
        print(f"DEBUG extract_session_features: session_features_to_merge columns: {session_features_to_merge.columns.tolist()}")
        df = df.merge(session_features_to_merge, on='_temp_row_id', how='left')
        print(f"DEBUG extract_session_features: After merging session_features_to_merge. df columns: {df.columns.tolist()}")

        # Fill any NaNs that might result from sessions with no activity or single events
        df['actions_per_session'] = df['actions_per_session'].fillna(0)
        df['distinct_actions_per_session'] = df['distinct_actions_per_session'].fillna(0)
        df['time_between_actions_in_session'] = df['time_between_actions_in_session'].fillna(0)
        # average_time_between_actions_in_session is already filled above

        # Add to feature columns
        self.numerical_columns.extend([
            'actions_per_session',
            'distinct_actions_per_session',
            'time_between_actions_in_session',
            'average_time_between_actions_in_session'
        ])

        return df
    
    def extract_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on regions."""
        self.logger.debug("Extracting region features")
        print(f"DEBUG extract_region_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_region_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_region_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure 'region' is a string and filled
        df['region_str'] = df['region'].fillna('unknown_region').astype(str)

        # Region frequency
        region_counts = df['region_str'].value_counts()
        df['region_frequency'] = df['region_str'].map(region_counts).fillna(0)

        # Top N regions as categorical features
        top_n_regions = region_counts.head(5).index.tolist() # Consider top 5 regions
        df['top_region'] = df['region_str'].apply(lambda x: x if x in top_n_regions else 'other_region')
        
        # Region changes per session - 'session_id' and 'region' are now guaranteed to exist.
        # Ensure 'session_id' is not NA for groupby. Fill with a placeholder string if needed
        df_temp_region = df.dropna(subset=['session_id', 'region_str', '_temp_row_id']).sort_values(by=['session_id', '_temp_row_id']).copy()
        print(f"DEBUG extract_region_features: df_temp_region columns after sort/copy: {df_temp_region.columns.tolist()}")
        print(f"DEBUG extract_region_features: df_temp_region index after sort/copy: {df_temp_region.index.name}")

        # Do not set index here; keep _temp_row_id as a column for merging
        df_temp_region['session_id_filled'] = df_temp_region['session_id'].fillna('unknown_session')
        region_changes_in_session_transformed = df_temp_region.groupby('session_id_filled')['region_str'].transform('nunique')
        # Assign directly to df_temp_region using _temp_row_id as index for alignment
        df_temp_region = df_temp_region.set_index('_temp_row_id')
        df_temp_region['region_changes_in_session'] = region_changes_in_session_transformed
        df_temp_region['region_changes_in_session'] = df_temp_region['region_changes_in_session'].fillna(0) # Fill NaN from transform
        
        print(f"DEBUG extract_region_features: Before merging region_changes_in_session. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp_region.reset_index()[['_temp_row_id', 'region_changes_in_session']], on='_temp_row_id', how='left').fillna({'region_changes_in_session': 0})
        print(f"DEBUG extract_region_features: After merging region_changes_in_session. df columns: {df.columns.tolist()}")

        # Add to feature columns
        self.numerical_columns.extend(['region_frequency', 'region_changes_in_session'])
        self.categorical_columns.append('top_region')
        
        # Drop the temporary column
        df.drop(columns=['region_str'], inplace=True)
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agents."""
        self.logger.debug("Extracting user agent features")
        print(f"DEBUG extract_user_agent_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_user_agent_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_user_agent_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure 'user_agent' is a string and filled
        df['user_agent_str'] = df['user_agent'].fillna('unknown_user_agent').astype(str)

        # User agent frequency
        ua_counts = df['user_agent_str'].value_counts()
        df['user_agent_frequency'] = df['user_agent_str'].map(ua_counts).fillna(0)
        
        # Top N user agents as categorical features
        top_n_uas = ua_counts.head(5).index.tolist() # Consider top 5 UAs
        df['top_user_agent'] = df['user_agent_str'].apply(lambda x: x if x in top_n_uas else 'other_user_agent')
        
        # Extract OS and browser from user agent string (simplified)
        # This is a basic parsing and might need more sophisticated libraries for production
        df['os'] = df['user_agent_str'].apply(lambda x: (
            'Windows' if 'Windows' in x else 
            'Mac' if 'Macintosh' in x else 
            'Linux' if 'Linux' in x else 
            'OtherOS'
        ))
        df['browser'] = df['user_agent_str'].apply(lambda x: (
            'Chrome' if 'Chrome' in x else 
            'Firefox' if 'Firefox' in x else 
            'Safari' if 'Safari' in x else 
            'OtherBrowser'
        ))
        
        # User agent changes per session - 'session_id' and 'user_agent' are now guaranteed to exist.
        # Ensure 'session_id' is not NA for groupby. Fill with a placeholder string if needed
        df_temp_ua = df.dropna(subset=['session_id', 'user_agent_str', '_temp_row_id']).sort_values(by=['session_id', 'timestamp', '_temp_row_id']).copy()
        print(f"DEBUG extract_user_agent_features: df_temp_ua columns after sort/copy: {df_temp_ua.columns.tolist()}")
        print(f"DEBUG extract_user_agent_features: df_temp_ua index after sort/copy: {df_temp_ua.index.name}")

        # Do not set index here; keep _temp_row_id as a column for merging
        df_temp_ua['session_id_filled'] = df_temp_ua['session_id'].fillna('unknown_session')
        user_agent_changes_in_session_transformed = df_temp_ua.groupby('session_id_filled')['user_agent_str'].transform('nunique')
        # Assign directly to df_temp_ua using _temp_row_id as index for alignment
        df_temp_ua = df_temp_ua.set_index('_temp_row_id')
        df_temp_ua['user_agent_changes_in_session'] = user_agent_changes_in_session_transformed
        df_temp_ua['user_agent_changes_in_session'] = df_temp_ua['user_agent_changes_in_session'].fillna(0)

        print(f"DEBUG extract_user_agent_features: Before merging user_agent_changes_in_session. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp_ua.reset_index()[['_temp_row_id', 'user_agent_changes_in_session']], on='_temp_row_id', how='left').fillna({'user_agent_changes_in_session': 0})
        print(f"DEBUG extract_user_agent_features: After merging user_agent_changes_in_session. df columns: {df.columns.tolist()}")

        # Add to feature columns
        self.numerical_columns.extend(['user_agent_frequency', 'user_agent_changes_in_session'])
        self.categorical_columns.extend(['top_user_agent', 'os', 'browser'])

        # Drop the temporary column
        df.drop(columns=['user_agent_str'], inplace=True)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Scales numerical features using StandardScaler."""
        if not self.numerical_columns:
            print("No numerical columns to scale.")
            return df

        # Ensure all numerical columns are present and valid before scaling
        cols_to_scale = [col for col in self.numerical_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        if not cols_to_scale:
            print("No valid numerical columns found for scaling after filtering.")
            return df

        # Fit scaler only on the first batch of data or if not fitted yet
        if is_training and not hasattr(self.scaler, 'mean_'):
            # Filter df to only numerical columns that are not all zeros or NaNs
            # to avoid issues with fitting on constant features
            train_data = df[cols_to_scale].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
            
            if not train_data.empty and train_data.nunique().sum() > 0: # Ensure there's actually varying data to fit on
                self.scaler.fit(train_data)
            else:
                print("Warning: No varying numerical data to fit scaler. Skipping scaling for this batch.")
                return df

        # Transform data. Handle potential new columns during transform by re-filtering.
        # Only transform if scaler has been fitted
        if hasattr(self.scaler, 'mean_'):
            scaled_data = self.scaler.transform(df[cols_to_scale])
            df[cols_to_scale] = scaled_data
            print(f"Scaled {len(cols_to_scale)} numerical features.")
        else:
            print("Scaler not fitted, skipping transformation.")
        
        return df

    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sequence-based features to detect unusual patterns of actions."""
        self.logger.debug("Extracting sequence features")
        print(f"DEBUG extract_sequence_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_sequence_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_sequence_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure required columns are present and properly formatted
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Sort by user and timestamp, including _temp_row_id for stability
        df_temp = df.dropna(subset=['timestamp', 'user_id', 'action', '_temp_row_id']).sort_values(['user_id', 'timestamp', '_temp_row_id']).copy()
        print(f"DEBUG extract_sequence_features: df_temp columns after sort/copy: {df_temp.columns.tolist()}")
        print(f"DEBUG extract_sequence_features: df_temp index after sort/copy: {df_temp.index.name}")
        # Do not set index here; keep _temp_row_id as a column for merging
        
        # Action sequence entropy (measures randomness in action patterns)
        def calculate_entropy(series):
            value_counts = series.value_counts(normalize=True)
            return -np.sum(value_counts * np.log2(value_counts)) if not value_counts.empty else 0
        
        # Calculate entropy for each user's action sequence
        action_entropy_calculated = df_temp.groupby('user_id')['action'].apply(calculate_entropy).rename('action_entropy')
        
        print(f"DEBUG extract_sequence_features: Before merging action_entropy. df columns: {df.columns.tolist()}")
        df_temp_entropy = df_temp.set_index('_temp_row_id')
        df_temp_entropy['action_entropy'] = df_temp_entropy['user_id'].map(action_entropy_calculated)
        df_temp_entropy['action_entropy'] = df_temp_entropy['action_entropy'].fillna(0)
        df = df.merge(df_temp_entropy.reset_index()[['_temp_row_id', 'action_entropy']], on='_temp_row_id', how='left').fillna({'action_entropy': 0})
        print(f"DEBUG extract_sequence_features: After merging action_entropy. df columns: {df.columns.tolist()}")

        # Action transition matrix features
        def get_transition_features(group):
            actions = group['action'].tolist()
            if len(actions) < 2: # Need at least two actions for a transition
                return pd.Series({'common_transition_prob': 0, 'unique_transitions': 0})
            transitions = list(zip(actions[:-1], actions[1:]))
            transition_counts = pd.Series(transitions).value_counts(normalize=True)
            return pd.Series({
                'common_transition_prob': transition_counts.max() if not transition_counts.empty else 0,
                'unique_transitions': len(transition_counts)
            })
        
        transition_features_calculated = df_temp.groupby('user_id').apply(get_transition_features)
        print(f"DEBUG extract_sequence_features: Before merging transition_features. df columns: {df.columns.tolist()}")
        df_temp_transition = df_temp.set_index('_temp_row_id')
        df_temp_transition['common_transition_prob'] = df_temp_transition['user_id'].map(transition_features_calculated['common_transition_prob'])
        df_temp_transition['unique_transitions'] = df_temp_transition['user_id'].map(transition_features_calculated['unique_transitions'])

        df = df.merge(df_temp_transition.reset_index()[['_temp_row_id', 'common_transition_prob', 'unique_transitions']], on='_temp_row_id', how='left').fillna(0)
        print(f"DEBUG extract_sequence_features: After merging transition_features. df columns: {df.columns.tolist()}")

        # Add to numerical columns
        self.numerical_columns.extend(['action_entropy', 'common_transition_prob', 'unique_transitions'])
        
        return df

    def extract_advanced_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced behavioral features focusing on unusual patterns."""
        self.logger.debug("Extracting advanced behavioral features")
        print(f"DEBUG extract_advanced_behavioral_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_advanced_behavioral_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_advanced_behavioral_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure required columns
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Calculate action velocity (actions per time unit)
        df_temp = df.dropna(subset=['timestamp', 'user_id', 'action', '_temp_row_id']).sort_values(['user_id', 'timestamp', '_temp_row_id']).copy() # Include _temp_row_id for stable sort
        print(f"DEBUG extract_advanced_behavioral_features: df_temp columns after sort/copy: {df_temp.columns.tolist()}")
        print(f"DEBUG extract_advanced_behavioral_features: df_temp index after sort/copy: {df_temp.index.name}")

        # Do not set index here; keep _temp_row_id as a column for merging
        df_temp['time_diff'] = df_temp.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df_temp['action_velocity'] = 1 / df_temp['time_diff'].replace(0, np.nan).fillna(1) # Avoid division by zero, fill with 1 for initial actions
        
        # Detect burst activity (sudden spikes in activity)
        # Use _temp_row_id as index for transform to ensure correct alignment upon assignment
        df_temp_burst = df_temp.set_index('_temp_row_id')
        df_temp_burst['rolling_velocity'] = df_temp_burst.groupby('user_id')['action_velocity'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df_temp_burst['velocity_std'] = df_temp_burst.groupby('user_id')['action_velocity'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        ).fillna(0) # Fill NaN for std where not enough periods
        
        # Handle cases where velocity_std might be zero to avoid division by zero in the next step
        # Replace 0 with a small epsilon to prevent division by zero in the burst calculation
        df_temp_burst['is_burst'] = ((df_temp_burst['action_velocity'] > df_temp_burst['rolling_velocity'] + 2 * df_temp_burst['velocity_std']) & (df_temp_burst['velocity_std'] > 0)).astype(int)
        
        # Resource access diversity
        df_temp_burst['resource_diversity'] = df_temp_burst.groupby('user_id')['resource'].transform(
            lambda x: x.nunique() / len(x) if len(x) > 0 else 0
        ).fillna(0)
        
        # Merge action_velocity, is_burst, resource_diversity back to original df using _temp_row_id
        print(f"DEBUG extract_advanced_behavioral_features: Before merging velocity/burst/diversity. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp_burst.reset_index()[['_temp_row_id', 'action_velocity', 'is_burst', 'resource_diversity']],
                      on='_temp_row_id', how='left').fillna({'action_velocity': 0, 'is_burst': 0, 'resource_diversity': 0})
        print(f"DEBUG extract_advanced_behavioral_features: After merging velocity/burst/diversity. df columns: {df.columns.tolist()}")

        # Action-resource co-occurrence patterns (limit to top N for efficiency)
        df_temp_cooc = df.dropna(subset=['user_id', 'action', 'resource', '_temp_row_id']).copy() # New temp df for co-occurrence
        df_temp_cooc['action_resource_pair'] = df_temp_cooc['action'] + '_' + df_temp_cooc['resource']
        
        # Calculate frequency of each pair
        pair_frequencies = df_temp_cooc['action_resource_pair'].value_counts(normalize=True)
        
        # Select top N most frequent pairs (e.g., top 20, adjust as needed)
        top_n_pairs = pair_frequencies.head(20).index.tolist()
        
        # Map all other pairs to an 'other_pair' category
        df_temp_cooc['action_resource_pair_reduced'] = df_temp_cooc['action_resource_pair'].apply(lambda x: x if x in top_n_pairs else 'other_pair')
        
        # Now, group by user_id and this reduced pair feature for counts
        pair_counts = df_temp_cooc.groupby(['user_id', 'action_resource_pair_reduced']).size().unstack(fill_value=0)
        pair_counts.columns = [f'pair_{col}' for col in pair_counts.columns] # Ensure unique column names
        print(f"DEBUG extract_advanced_behavioral_features: Before merging pair_counts. df columns: {df.columns.tolist()}")
        df = df.merge(pair_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_advanced_behavioral_features: After merging pair_counts. df columns: {df.columns.tolist()}")
        
        # Add to numerical columns
        self.numerical_columns.extend([
            'action_velocity', 'is_burst', 'resource_diversity'
        ] + pair_counts.columns.tolist())
        
        return df

    def extract_temporal_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced temporal pattern features."""
        self.logger.debug("Extracting temporal pattern features")
        print(f"DEBUG extract_temporal_pattern_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_temporal_pattern_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_temporal_pattern_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        
        # Calculate time-based patterns
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # Time-of-day patterns
        df['is_early_morning'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        
        # Calculate user's typical active hours
        user_active_hours = df.groupby('user_id')['hour'].agg(['mean', 'std']).fillna(0)
        # Rename columns to avoid conflict during merge
        user_active_hours.columns = ['user_mean_hour', 'user_std_hour']
        print(f"DEBUG extract_temporal_pattern_features: Before merging user_active_hours. df columns: {df.columns.tolist()}")
        df = df.merge(user_active_hours.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_temporal_pattern_features: After merging user_active_hours. df columns: {df.columns.tolist()}")

        # Optimized calculation for hours_from_mean
        df['hours_from_mean'] = np.abs(df['hour'] - df['user_mean_hour'])
        
        # Session timing patterns
        # Ensure 'session_id', 'timestamp' and '_temp_row_id' are not null for these calculations
        df_temp = df.dropna(subset=['session_id', 'timestamp', 'hour', '_temp_row_id']).sort_values(by=['session_id', 'timestamp', '_temp_row_id']).copy()
        print(f"DEBUG extract_temporal_pattern_features: df_temp columns after sort/copy: {df_temp.columns.tolist()}")
        print(f"DEBUG extract_temporal_pattern_features: df_temp index after sort/copy: {df_temp.index.name}")

        # Do not set index here; keep _temp_row_id as a column for merging
        session_start_hour_transformed = df_temp.groupby('session_id')['hour'].transform('first')
        session_end_hour_transformed = df_temp.groupby('session_id')['hour'].transform('last')
        # Assign to df_temp using _temp_row_id as index for alignment
        df_temp = df_temp.set_index('_temp_row_id')
        df_temp['session_start_hour'] = session_start_hour_transformed
        df_temp['session_end_hour'] = session_end_hour_transformed
        df_temp['session_hour_span'] = df_temp['session_end_hour'] - df_temp['session_start_hour']

        # Merge session-based temporal features back to the original df
        print(f"DEBUG extract_temporal_pattern_features: Before merging session_timing. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp.reset_index()[['_temp_row_id', 'session_start_hour', 'session_end_hour', 'session_hour_span']],
                      on='_temp_row_id', how='left').fillna(0)
        print(f"DEBUG extract_temporal_pattern_features: After merging session_timing. df columns: {df.columns.tolist()}")

        # Drop the temporary mean/std columns after calculation
        df.drop(columns=['user_mean_hour', 'user_std_hour'], errors='ignore', inplace=True)
        
        # Add to numerical columns
        self.numerical_columns.extend([
            'is_early_morning', 'is_morning', 'is_afternoon', 'is_evening',
            'hours_from_mean', 'session_hour_span'
        ])
        
        return df

    def extract_cyberark_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features specific to CyberArk logs."""
        self.logger.debug("Extracting cyberark features")
        print(f"DEBUG extract_cyberark_features: DF columns at start: {df.columns.tolist()}")
        print(f"DEBUG extract_cyberark_features: DF index at start: {df.index.name}")
        print(f"DEBUG extract_cyberark_features: '_temp_row_id' in DF columns at start: {'_temp_row_id' in df.columns}")

        # Ensure required CyberArk columns exist and are handled
        required_cyberark_cols = [
            'privileged_account_used',
            'vault_name',
            'session_duration_seconds',
            'is_privileged_session',
            'policy_violation',
            'reason_for_access',
            'ticket_id'
        ]
        for col in required_cyberark_cols:
            if col not in df.columns:
                df[col] = np.nan # Add column if missing, fill with NaN
            # Convert to string for categorical/grouping where appropriate, fillna for consistency
            if df[col].dtype == 'object' or col in ['privileged_account_used', 'vault_name', 'reason_for_access', 'ticket_id']:
                df[col] = df[col].fillna(f'unknown_{col}').astype(str)
            elif col in ['is_privileged_session', 'policy_violation']:
                df[col] = df[col].fillna(False).astype(bool)
            elif col == 'session_duration_seconds':
                df[col] = df[col].fillna(0).astype(float)
        
        # --- Privileged Account Usage Features ---
        # Frequency of each privileged account used per user
        priv_acc_counts = df.groupby(['user_id', 'privileged_account_used']).size().unstack(fill_value=0)
        priv_acc_counts.columns = [f'priv_acc_{col}_count' for col in priv_acc_counts.columns]
        print(f"DEBUG extract_cyberark_features: Before merging priv_acc_counts. df columns: {df.columns.tolist()}")
        df = df.merge(priv_acc_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_cyberark_features: After merging priv_acc_counts. df columns: {df.columns.tolist()}")

        self.numerical_columns.extend(priv_acc_counts.columns.tolist())

        # Top N privileged accounts as categorical
        top_n_priv_acc = df['privileged_account_used'].value_counts().head(10).index.tolist()
        df['top_priv_account'] = df['privileged_account_used'].apply(lambda x: x if x in top_n_priv_acc else 'other_priv_account')
        self.categorical_columns.append('top_priv_account')

        # --- Vault Access Features ---
        # Frequency of each vault accessed per user
        vault_counts = df.groupby(['user_id', 'vault_name']).size().unstack(fill_value=0)
        vault_counts.columns = [f'vault_{col}_count' for col in vault_counts.columns]
        print(f"DEBUG extract_cyberark_features: Before merging vault_counts. df columns: {df.columns.tolist()}")
        df = df.merge(vault_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_cyberark_features: After merging vault_counts. df columns: {df.columns.tolist()}")

        self.numerical_columns.extend(vault_counts.columns.tolist())

        # Top N vaults as categorical
        top_n_vaults = df['vault_name'].value_counts().head(5).index.tolist()
        df['top_vault'] = df['vault_name'].apply(lambda x: x if x in top_n_vaults else 'other_vault')
        self.categorical_columns.append('top_vault')

        # --- Session Duration Features ---
        # Min, Max, Mean, Std of session duration per user (for privileged sessions)
        session_duration_stats = df[df['is_privileged_session']].groupby('user_id')['session_duration_seconds'].agg([
            'min', 'max', 'mean', 'std'
        ]).add_prefix('priv_session_duration_').fillna(0) # Fill NaN for users with no privileged sessions
        print(f"DEBUG extract_cyberark_features: Before merging session_duration_stats. df columns: {df.columns.tolist()}")
        df = df.merge(session_duration_stats.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_cyberark_features: After merging session_duration_stats. df columns: {df.columns.tolist()}")

        # Flag for unusually long/short privileged sessions (e.g., > 3 std from mean, if mean/std exist)
        df['is_long_priv_session'] = 0
        df['is_short_priv_session'] = 0

        # Only apply this logic if std dev is not zero and mean is not zero
        if 'priv_session_duration_mean' in df.columns and 'priv_session_duration_std' in df.columns:
            df.loc[df['is_privileged_session'] == True, 'is_long_priv_session'] = (
                df['session_duration_seconds'] > (df['priv_session_duration_mean'] + 3 * df['priv_session_duration_std'])).astype(int)
            df.loc[df['is_privileged_session'] == True, 'is_short_priv_session'] = (
                df['session_duration_seconds'] < (df['priv_session_duration_mean'] - 3 * df['priv_session_duration_std'])).astype(int)

        self.numerical_columns.extend(['is_long_priv_session', 'is_short_priv_session'])

        # --- Policy Violation Features ---
        # Count of policy violations per user
        policy_violation_counts = df.groupby('user_id')['policy_violation'].sum().rename('policy_violation_count')
        print(f"DEBUG extract_cyberark_features: Before merging policy_violation_counts. df columns: {df.columns.tolist()}")
        df = df.merge(policy_violation_counts.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_cyberark_features: After merging policy_violation_counts. df columns: {df.columns.tolist()}")

        # Ratio of policy violations to total actions per user
        total_actions_per_user = df.groupby('user_id').size().rename('total_actions_count')
        print(f"DEBUG extract_cyberark_features: Before merging total_actions_per_user. df columns: {df.columns.tolist()}")
        df = df.merge(total_actions_per_user.reset_index(), on='user_id', how='left').fillna(0)
        print(f"DEBUG extract_cyberark_features: After merging total_actions_per_user. df columns: {df.columns.tolist()}")

        df['policy_violation_ratio'] = df['policy_violation_count'] / df['total_actions_count']
        df['policy_violation_ratio'] = df['policy_violation_ratio'].fillna(0) # Fill NaN if total_actions_count is 0
        self.numerical_columns.append('policy_violation_ratio')
        self.numerical_columns.append('is_privileged_session') # Add this as a direct numerical feature (0 or 1)

        # --- Reason for Access & Ticket ID Features ---
        # Use a df_temp to ensure _temp_row_id is preserved for merging
        df_temp_reason = df.dropna(subset=['is_privileged_session', 'reason_for_access', '_temp_row_id']).copy()
        df_temp_reason['missing_reason_for_access'] = ((df_temp_reason['is_privileged_session'] == True) & 
                                           (df_temp_reason['reason_for_access'] == 'unknown_reason_for_access')).astype(int)
        print(f"DEBUG extract_cyberark_features: Before merging missing_reason_for_access. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp_reason[['_temp_row_id', 'missing_reason_for_access']], on='_temp_row_id', how='left').fillna({'missing_reason_for_access': 0})
        print(f"DEBUG extract_cyberark_features: After merging missing_reason_for_access. df columns: {df.columns.tolist()}")

        self.numerical_columns.append('missing_reason_for_access')

        # Flag for missing ticket ID for privileged sessions
        df_temp_ticket = df.dropna(subset=['is_privileged_session', 'ticket_id', '_temp_row_id']).copy()
        df_temp_ticket['missing_ticket_id'] = ((df_temp_ticket['is_privileged_session'] == True) & 
                                   (df_temp_ticket['ticket_id'] == 'unknown_ticket_id')).astype(int)
        print(f"DEBUG extract_cyberark_features: Before merging missing_ticket_id. df columns: {df.columns.tolist()}")
        df = df.merge(df_temp_ticket[['_temp_row_id', 'missing_ticket_id']], on='_temp_row_id', how='left').fillna({'missing_ticket_id': 0})
        print(f"DEBUG extract_cyberark_features: After merging missing_ticket_id. df columns: {df.columns.tolist()}")

        self.numerical_columns.append('missing_ticket_id')

        return df

    def engineer_features(self, df: pd.DataFrame, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        """
        Engineer features from the IAM logs.
        
        Args:
            df (pd.DataFrame): Input DataFrame with raw log data
            progress_callback (callable, optional): Function to report progress
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering")
        if df.empty:
            self.logger.warning("Empty DataFrame received for feature engineering")
            return df
            
        # Store original columns for reference
        original_columns = df.columns.tolist()
        self.logger.debug(f"Original columns: {original_columns}")
        
        # Create a copy to avoid modifying the original
        df_features = df.copy()
        
        # 1. Time-based features
        if progress_callback:
            progress_callback(1, 7, "Extracting time features...")
        df_features = self.extract_time_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After time features, columns: {df_features.columns.tolist()}")

        # 2. IP-based features
        if progress_callback:
            progress_callback(2, 7, "Extracting IP features...")
        df_features = self.extract_ip_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After IP features, columns: {df_features.columns.tolist()}")

        # 3. Behavioral features
        if progress_callback:
            progress_callback(3, 7, "Extracting behavioral features...")
        df_features = self.extract_behavioral_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After behavioral features, columns: {df_features.columns.tolist()}")

        # 4. Session features
        if progress_callback:
            progress_callback(4, 7, "Extracting session features...")
        df_features = self.extract_session_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After session features, columns: {df_features.columns.tolist()}")

        # 5. Region features
        if progress_callback:
            progress_callback(5, 7, "Extracting region features...")
        df_features = self.extract_region_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After region features, columns: {df_features.columns.tolist()}")

        # 6. User Agent features
        if progress_callback:
            progress_callback(6, 7, "Extracting user agent features...")
        df_features = self.extract_user_agent_features(df_features)
        self.logger.debug(f"DEBUG: engineer_features - After user agent features, columns: {df_features.columns.tolist()}")

        # 7. CyberArk specific features (if applicable, based on 'log_type' which should be in raw logs)
        if progress_callback:
            progress_callback(7, 7, "Extracting CyberArk specific features...")
        df_features = self.extract_cyberark_features(df_features) # Handles if log_type is not present
        self.logger.debug(f"DEBUG: engineer_features - After CyberArk features, columns: {df_features.columns.tolist()}")

        # Store the feature columns
        self.feature_columns = [col for col in df_features.columns if col not in original_columns]
        self.logger.info(f"Generated {len(self.feature_columns)} features")
        self.logger.debug(f"Feature columns: {self.feature_columns}")
        
        if progress_callback:
            progress_callback(7, 7, "Feature engineering complete!")
            
        return df_features

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
    print(f"Total number of features: {len(engineer.numerical_columns)}")
    print("\nFeature columns:")
    print(engineer.numerical_columns)
    
    print("\nSample of engineered data:")
    print(df_with_features[engineer.numerical_columns].head()) 