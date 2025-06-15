import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Callable
from datetime import datetime, timedelta
import ipaddress
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
        self.categorical_columns = []
        self.numerical_columns = []
        self.scaler = StandardScaler()
        
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamps."""
        # Convert timestamp to datetime if it's not already
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Basic time features
        # Use original df directly as _temp_row_id is already set
        df_temp = df.dropna(subset=['timestamp']).copy()

        df['hour'] = df_temp['timestamp'].dt.hour
        df['day_of_week'] = df_temp['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_working_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Add cyclical time features for hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'].fillna(0)/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'].fillna(0)/24)
        
        # Time since last access for each user - ensure sorting includes _temp_row_id
        df = df.sort_values(['user_id', 'timestamp', '_temp_row_id'])
        # Ensure user_id is not NA for groupby. Fill with a placeholder string if needed
        df['user_id_filled'] = df['user_id'].fillna('unknown_user')
        df['time_since_last_access'] = df.groupby('user_id_filled')['timestamp'].diff().dt.total_seconds()
        df['time_since_last_access'] = df['time_since_last_access'].fillna(0)
        df.drop(columns=['user_id_filled'], inplace=True)

        # Session-based features - 'session_start' and 'session_end' are now guaranteed to exist
        # and are datetime objects or NaT from IAMLogReader. Fill NaT values for calculation.
        df['session_duration'] = (df['session_end'].fillna(df['timestamp']) - df['session_start'].fillna(df['timestamp'])).dt.total_seconds()
        df['session_duration'] = df['session_duration'].fillna(0) # Fill NaN from NaT difference with 0
            
        # Add to feature columns
        self.numerical_columns.extend([
            'hour', 'day_of_week', 'is_weekend', 'is_working_hour',
            'time_since_last_access', 'hour_sin', 'hour_cos', 'session_duration'
        ])
        
        return df
    
    def extract_ip_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from IP addresses."""
        def is_private_ip(ip):
            try:
                # Only attempt conversion if ip is a string and not 'unknown' or empty
                if isinstance(ip, str) and ip not in ('unknown', ''):
                    return ipaddress.ip_address(ip).is_private
                return False # Treat non-IPs or 'unknown' as not private
            except ValueError: # Catches invalid IP formats
                return False
            except Exception as e:
                # Log other unexpected errors but return False
                # print(f"Error checking private IP for {ip}: {e}") # Temporarily remove debug print
                return False
        
        # IP type features - apply only to non-null and non-'unknown' IP addresses
        # Convert to string and fillna, then apply.
        df['ip_address_str'] = df['ip_address'].astype(str).fillna('unknown')
        df['is_private_ip'] = df['ip_address_str'].apply(is_private_ip).astype(int)
        
        # IP frequency features
        # Value counts will include 'unknown' which is acceptable for frequency
        ip_counts = df['ip_address_str'].value_counts()
        df['ip_frequency'] = df['ip_address_str'].map(ip_counts).fillna(0) # Fillna for IPs not in counts (shouldn't happen if all are in df)

        # Top N IP addresses as categorical features
        # Ensure 'ip_address_str' is used here as well
        top_n_ips = ip_counts.head(20).index.tolist() # Consider top 20 IPs
        df['top_ip'] = df['ip_address_str'].apply(lambda x: x if x in top_n_ips else 'other_ip')
        
        # IP changes per session - 'session_id' and 'ip_address' are now guaranteed to exist.
        # Ensure 'session_id' is not NA for groupby. Fill with a placeholder string if needed
        df_temp = df.dropna(subset=['session_id', 'ip_address_str', '_temp_row_id']).sort_values(by=['session_id', 'timestamp', '_temp_row_id']).copy()
        df_temp = df_temp.set_index('_temp_row_id') # Set _temp_row_id as index
        df_temp['session_id_filled'] = df_temp['session_id'].fillna('unknown_session')
        df_temp['ip_changes_in_session'] = df_temp.groupby('session_id_filled')['ip_address_str'].transform('nunique')
        df_temp['ip_changes_in_session'] = df_temp['ip_changes_in_session'].fillna(0) # Fill NaN from transform with 0
        
        df = df.merge(df_temp.reset_index()[['_temp_row_id', 'ip_changes_in_session']], on='_temp_row_id', how='left').fillna({'ip_changes_in_session': 0})

        # Add to feature columns
        self.numerical_columns.extend(['is_private_ip', 'ip_frequency', 'ip_changes_in_session'])
        self.categorical_columns.append('top_ip')
        
        # Drop the temporary column
        df.drop(columns=['ip_address_str'], inplace=True)
        
        return df
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features based on user actions."""
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
        df_temp_behavioral = df.sort_values(by=['user_id', 'timestamp', '_temp_row_id']).copy()
        df_temp_behavioral = df_temp_behavioral.set_index('_temp_row_id') # Set _temp_row_id as index
        print(f"DEBUG: Behavioral Features - After df_temp_behavioral set_index: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")

        # Actions per minute
        print(f"DEBUG: Behavioral Features - Before actions_per_minute rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        actions_per_minute_rolled = df_temp_behavioral.groupby('user_id').rolling('1min', on='timestamp')['action'].count().reset_index()
        actions_per_minute_df = actions_per_minute_rolled.rename(columns={'action': 'actions_per_minute'})
        print(f"DEBUG: extract_behavioral_features - df.index.is_unique before actions_per_minute merge: {df.index.is_unique}") # DEBUG PRINT
        print(f"DEBUG: Behavioral Features - actions_per_minute_df columns: {actions_per_minute_df.columns.tolist()}")
        df = df.merge(actions_per_minute_df[['_temp_row_id', 'actions_per_minute']], on='_temp_row_id', how='left').fillna({'actions_per_minute': 0})
        print(f"DEBUG: Behavioral Features - After actions_per_minute merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Unique actions per hour
        print(f"DEBUG: Behavioral Features - Before unique_actions_per_hour rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        unique_actions_per_hour_rolled = df_temp_behavioral.groupby('user_id').rolling('1h', on='timestamp')['action'].apply(lambda x: x.nunique(), raw=False).reset_index()
        unique_actions_per_hour_df = unique_actions_per_hour_rolled.rename(columns={'action': 'unique_actions_per_hour'})
        print(f"DEBUG: Behavioral Features - unique_actions_per_hour_df columns: {unique_actions_per_hour_df.columns.tolist()}")
        df = df.merge(unique_actions_per_hour_df[['_temp_row_id', 'unique_actions_per_hour']], on='_temp_row_id', how='left').fillna({'unique_actions_per_hour': 0})
        print(f"DEBUG: Behavioral Features - After unique_actions_per_hour merge: Columns={df.columns.tolist()}, Index={df.index.name}")

        # Unique resources per hour
        print(f"DEBUG: Behavioral Features - Before unique_resources_per_hour rolling: Columns={df_temp_behavioral.columns.tolist()}, Index={df_temp_behavioral.index.name}")
        unique_resources_per_hour_rolled = df_temp_behavioral.groupby('user_id').rolling('1h', on='timestamp')['resource'].apply(lambda x: x.nunique(), raw=False).reset_index()
        unique_resources_per_hour_df = unique_resources_per_hour_rolled.rename(columns={'resource': 'unique_resources_per_hour'})
        print(f"DEBUG: Behavioral Features - unique_resources_per_hour_df columns: {unique_resources_per_hour_df.columns.tolist()}")
        df = df.merge(unique_resources_per_hour_df[['_temp_row_id', 'unique_resources_per_hour']], on='_temp_row_id', how='left').fillna({'unique_resources_per_hour': 0})
        print(f"DEBUG: Behavioral Features - After unique_resources_per_hour merge: Columns={df.columns.tolist()}, Index={df.index.name}")

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
        # Ensure 'session_id', 'resource', 'timestamp', 'action' are filled and correctly typed
        df['session_id'] = df['session_id'].fillna('unknown_session').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        # Ensure timestamp is datetime, already done by IAMLogReader standardization, but fill NaT for calculation robustness
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows where timestamp is NaT as they are crucial for time-based calculations
        # Include _temp_row_id in sorting to maintain its relative order within groups if timestamps are identical
        df_temp = df.dropna(subset=['timestamp']).sort_values(by=['session_id', 'timestamp', '_temp_row_id']).copy()
        df_temp = df_temp.set_index('_temp_row_id') # Set _temp_row_id as index

        # Only process sessions if there are valid session IDs beyond 'unknown_session'
        # and if there's enough data to form groups (i.e., more than one record after dropping NaNs)
        if not df_temp[df_temp['session_id'] != 'unknown_session'].empty and len(df_temp) > 0:

            # 1. Actions per session
            df_temp['actions_per_session'] = df_temp.groupby('session_id')['session_id'].transform('size')

            # 2. Distinct actions per session
            df_temp['distinct_actions_per_session'] = df_temp.groupby('session_id')['action'].transform('nunique')

            # 3. Time between actions in session (for each user within a session)
            df_temp['time_between_actions_in_session'] = df_temp.groupby('session_id')['timestamp'].diff().dt.total_seconds().fillna(0)

            # 4. Average time between actions in session (Optimized using rolling mean)
            df_temp['average_time_between_actions_in_session'] = df_temp.groupby('session_id')['time_between_actions_in_session'].rolling(window=5, min_periods=1).mean()
            df_temp['average_time_between_actions_in_session'] = df_temp['average_time_between_actions_in_session'].fillna(0)

            # Merge back to the original DataFrame using _temp_row_id
            features_to_merge_cols = [
                'actions_per_session',
                'distinct_actions_per_session',
                'time_between_actions_in_session',
                'average_time_between_actions_in_session',
                # _temp_row_id will now be part of the index after reset_index()
            ]
            # Filter df_temp to only the columns needed for merging and reset its index to make _temp_row_id a column
            features_to_merge = df_temp[features_to_merge_cols].reset_index() # Reset index here
            df = df.merge(features_to_merge, on='_temp_row_id', how='left')

            # Fill any NaNs that might result from sessions with no activity or single events
            df['actions_per_session'] = df['actions_per_session'].fillna(0)
            df['distinct_actions_per_session'] = df['distinct_actions_per_session'].fillna(0)
            df['time_between_actions_in_session'] = df['time_between_actions_in_session'].fillna(0)
            df['average_time_between_actions_in_session'] = df['average_time_between_actions_in_session'].fillna(0)

            # Add to feature columns
            self.numerical_columns.extend([
                'actions_per_session',
                'distinct_actions_per_session',
                'time_between_actions_in_session',
                'average_time_between_actions_in_session'
            ])
        else:
            # If no valid sessions or not enough data to process, create empty columns with default values
            df['actions_per_session'] = 0.0
            df['distinct_actions_per_session'] = 0.0
            df['time_between_actions_in_session'] = 0.0
            df['average_time_between_actions_in_session'] = 0.0
            # Add to feature columns - these will be present, but perhaps not useful if always 0
            self.numerical_columns.extend([
                'actions_per_session',
                'distinct_actions_per_session',
                'time_between_actions_in_session',
                'average_time_between_actions_in_session'
            ])

        return df
    
    def extract_region_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features based on regions."""
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
        df_temp_region = df.dropna(subset=['session_id', 'region_str']).sort_values(by=['session_id', '_temp_row_id']).copy()
        df_temp_region['session_id_filled'] = df_temp_region['session_id'].fillna('unknown_session')
        df_temp_region['region_changes_in_session'] = df_temp_region.groupby('session_id_filled')['region_str'].transform('nunique')
        df_temp_region['region_changes_in_session'] = df_temp_region['region_changes_in_session'].fillna(0) # Fill NaN from transform
        
        df = df.merge(df_temp_region[['_temp_row_id', 'region_changes_in_session']], on='_temp_row_id', how='left').fillna({'region_changes_in_session': 0})

        # Add to feature columns
        self.numerical_columns.extend(['region_frequency', 'region_changes_in_session'])
        self.categorical_columns.append('top_region')
        
        # Drop the temporary column
        df.drop(columns=['region_str'], inplace=True)
        
        return df
    
    def extract_user_agent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from user agents."""
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
        df_temp_ua = df_temp_ua.set_index('_temp_row_id') # Set _temp_row_id as index
        df_temp_ua['session_id_filled'] = df_temp_ua['session_id'].fillna('unknown_session')
        df_temp_ua['user_agent_changes_in_session'] = df_temp_ua.groupby('session_id_filled')['user_agent_str'].transform('nunique')
        df_temp_ua['user_agent_changes_in_session'] = df_temp_ua['user_agent_changes_in_session'].fillna(0)

        df = df.merge(df_temp_ua.reset_index()[['_temp_row_id', 'user_agent_changes_in_session']], on='_temp_row_id', how='left').fillna({'user_agent_changes_in_session': 0})

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
        # Ensure required columns are present and properly formatted
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Sort by user and timestamp, including _temp_row_id for stability
        df_temp = df.sort_values(['user_id', 'timestamp', '_temp_row_id']).copy()
        df_temp = df_temp.set_index('_temp_row_id') # Set _temp_row_id as index
        
        # Action sequence entropy (measures randomness in action patterns)
        def calculate_entropy(series):
            value_counts = series.value_counts(normalize=True)
            return -np.sum(value_counts * np.log2(value_counts)) if not value_counts.empty else 0
        
        # Calculate entropy for each user's action sequence
        action_entropy = df_temp.groupby('user_id')['action'].apply(calculate_entropy).rename('action_entropy').reset_index()
        df = df.merge(action_entropy, on='user_id', how='left').fillna({'action_entropy': 0})
        
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
        
        transition_features = df_temp.groupby('user_id').apply(get_transition_features).reset_index()
        df = df.merge(transition_features, on='user_id', how='left').fillna(0)
        
        # Add to numerical columns
        self.numerical_columns.extend(['action_entropy', 'common_transition_prob', 'unique_transitions'])
        
        return df

    def extract_advanced_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced behavioral features focusing on unusual patterns."""
        # Ensure required columns
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Calculate action velocity (actions per time unit)
        df_temp = df.sort_values(['user_id', 'timestamp', '_temp_row_id']).copy() # Include _temp_row_id for stable sort
        df_temp = df_temp.set_index('_temp_row_id') # Set _temp_row_id as index
        df_temp['time_diff'] = df_temp.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df_temp['action_velocity'] = 1 / df_temp['time_diff'].replace(0, np.nan).fillna(1) # Avoid division by zero, fill with 1 for initial actions
        
        # Detect burst activity (sudden spikes in activity)
        df_temp['rolling_velocity'] = df_temp.groupby('user_id')['action_velocity'].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df_temp['velocity_std'] = df_temp.groupby('user_id')['action_velocity'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        ).fillna(0) # Fill NaN for std where not enough periods
        
        # Handle cases where velocity_std might be zero to avoid division by zero in the next step
        # Replace 0 with a small epsilon to prevent division by zero in the burst calculation
        df_temp['is_burst'] = ((df_temp['action_velocity'] > df_temp['rolling_velocity'] + 2 * df_temp['velocity_std']) & (df_temp['velocity_std'] > 0)).astype(int)
        
        # Resource access diversity
        df_temp['resource_diversity'] = df_temp.groupby('user_id')['resource'].transform(
            lambda x: x.nunique() / len(x) if len(x) > 0 else 0
        ).fillna(0)
        
        # Merge action_velocity, is_burst, resource_diversity back to original df using _temp_row_id
        # Filter df_temp to only columns needed for merge, including _temp_row_id
        df = df.merge(df_temp.reset_index()[['_temp_row_id', 'action_velocity', 'is_burst', 'resource_diversity']],
                      on='_temp_row_id', how='left').fillna({'action_velocity': 0, 'is_burst': 0, 'resource_diversity': 0})

        # Action-resource co-occurrence patterns (limit to top N for efficiency)
        df_temp['action_resource_pair'] = df_temp['action'] + '_' + df_temp['resource']
        
        # Calculate frequency of each pair
        pair_frequencies = df_temp['action_resource_pair'].value_counts(normalize=True)
        
        # Select top N most frequent pairs (e.g., top 20, adjust as needed)
        top_n_pairs = pair_frequencies.head(20).index.tolist()
        
        # Map all other pairs to an 'other_pair' category
        df_temp['action_resource_pair_reduced'] = df_temp['action_resource_pair'].apply(lambda x: x if x in top_n_pairs else 'other_pair')
        
        # Now, group by user_id and this reduced pair feature for counts
        pair_counts = df_temp.groupby(['user_id', 'action_resource_pair_reduced']).size().unstack(fill_value=0)
        pair_counts.columns = [f'pair_{col}' for col in pair_counts.columns] # Ensure unique column names
        df = df.merge(pair_counts.reset_index(), on='user_id', how='left').fillna(0)
        
        # Add to numerical columns
        self.numerical_columns.extend([
            'action_velocity', 'is_burst', 'resource_diversity'
        ] + pair_counts.columns.tolist())
        
        return df

    def extract_temporal_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract advanced temporal pattern features."""
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
        df = df.merge(user_active_hours.reset_index(), on='user_id', how='left').fillna(0)

        # Optimized calculation for hours_from_mean
        df['hours_from_mean'] = np.abs(df['hour'] - df['user_mean_hour'])
        
        # Session timing patterns
        # Ensure 'session_id', 'timestamp' and '_temp_row_id' are not null for these calculations
        df_temp = df.dropna(subset=['session_id', 'timestamp', 'hour', '_temp_row_id']).copy()
        df_temp = df_temp.set_index('_temp_row_id') # Set _temp_row_id as index
        df_temp['session_start_hour'] = df_temp.groupby('session_id')['hour'].transform('first')
        df_temp['session_end_hour'] = df_temp.groupby('session_id')['hour'].transform('last')
        df_temp['session_hour_span'] = df_temp['session_end_hour'] - df_temp['session_start_hour']

        # Merge session-based temporal features back to the original df
        # Reset index to make _temp_row_id a column for merging
        df = df.merge(df_temp[['session_id', 'timestamp', 'session_start_hour', 'session_end_hour', 'session_hour_span']].reset_index(),
                      on=['session_id', 'timestamp', '_temp_row_id'], how='left').fillna(0)

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
        df = df.merge(priv_acc_counts.reset_index(), on='user_id', how='left').fillna(0)
        self.numerical_columns.extend(priv_acc_counts.columns.tolist())

        # Top N privileged accounts as categorical
        top_n_priv_acc = df['privileged_account_used'].value_counts().head(10).index.tolist()
        df['top_priv_account'] = df['privileged_account_used'].apply(lambda x: x if x in top_n_priv_acc else 'other_priv_account')
        self.categorical_columns.append('top_priv_account')

        # --- Vault Access Features ---
        # Frequency of each vault accessed per user
        vault_counts = df.groupby(['user_id', 'vault_name']).size().unstack(fill_value=0)
        vault_counts.columns = [f'vault_{col}_count' for col in vault_counts.columns]
        df = df.merge(vault_counts.reset_index(), on='user_id', how='left').fillna(0)
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
        df = df.merge(session_duration_stats.reset_index(), on='user_id', how='left').fillna(0)
        self.numerical_columns.extend(session_duration_stats.columns.tolist())

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
        df = df.merge(policy_violation_counts.reset_index(), on='user_id', how='left').fillna(0)
        self.numerical_columns.append('policy_violation_count')

        # Ratio of policy violations to total actions per user
        total_actions_per_user = df.groupby('user_id').size().rename('total_actions_count')
        df = df.merge(total_actions_per_user.reset_index(), on='user_id', how='left').fillna(0)
        df['policy_violation_ratio'] = df['policy_violation_count'] / df['total_actions_count']
        df['policy_violation_ratio'] = df['policy_violation_ratio'].fillna(0) # Fill NaN if total_actions_count is 0
        self.numerical_columns.append('policy_violation_ratio')
        self.numerical_columns.append('is_privileged_session') # Add this as a direct numerical feature (0 or 1)

        # --- Reason for Access & Ticket ID Features ---
        # Flag for missing reason for access for privileged sessions
        # Use a df_temp to ensure _temp_row_id is preserved for merging
        df_temp_reason = df.copy()
        df_temp_reason = df_temp_reason.set_index('_temp_row_id')
        df_temp_reason['missing_reason_for_access'] = ((df_temp_reason['is_privileged_session'] == True) & 
                                           (df_temp_reason['reason_for_access'] == 'unknown_reason_for_access')).astype(int)
        df = df.merge(df_temp_reason.reset_index()[['_temp_row_id', 'missing_reason_for_access']], on='_temp_row_id', how='left').fillna({'missing_reason_for_access': 0})
        self.numerical_columns.append('missing_reason_for_access')

        # Flag for missing ticket ID for privileged sessions
        df_temp_ticket = df.copy()
        df_temp_ticket = df_temp_ticket.set_index('_temp_row_id')
        df_temp_ticket['missing_ticket_id'] = ((df_temp_ticket['is_privileged_session'] == True) & 
                                   (df_temp_ticket['ticket_id'] == 'unknown_ticket_id')).astype(int)
        df = df.merge(df_temp_ticket.reset_index()[['_temp_row_id', 'missing_ticket_id']], on='_temp_row_id', how='left').fillna({'missing_ticket_id': 0})
        self.numerical_columns.append('missing_ticket_id')

        return df

    def engineer_features(self, df: pd.DataFrame, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        """Engineer all features for the dataset."""
        def update_progress(message):
            if progress_callback:
                progress_callback(0, 100, message)
        
        # Reset feature columns
        self.feature_columns = []
        self.categorical_columns = [] # Reset for each run
        self.numerical_columns = []
        
        # Create a stable unique row ID at the beginning
        df['_temp_row_id'] = df.index 
        print(f"DEBUG: engineer_features - df.index.is_unique at start: {df.index.is_unique}") # DEBUG PRINT

        # Identify initial categorical columns that should be one-hot encoded
        # Assuming 'role' is a primary categorical feature not converted to numerical counts elsewhere
        initial_categorical_cols = [col for col in ['role'] if col in df.columns]
        self.categorical_columns.extend(initial_categorical_cols)

        # Apply all feature engineering steps
        update_progress("Extracting time features...")
        df = self.extract_time_features(df)
        
        update_progress("Extracting IP features...")
        df = self.extract_ip_features(df)
        
        update_progress("Extracting behavioral features...")
        df = self.extract_behavioral_features(df)
        
        update_progress("Extracting session features...")
        df = self.extract_session_features(df)
        
        update_progress("Extracting region features...")
        df = self.extract_region_features(df)
        
        update_progress("Extracting user agent features...")
        df = self.extract_user_agent_features(df)
        
        # Add new feature extraction steps
        update_progress("Extracting sequence features...")
        df = self.extract_sequence_features(df)
        
        update_progress("Extracting advanced behavioral features...")
        df = self.extract_advanced_behavioral_features(df)
        
        update_progress("Extracting temporal pattern features...")
        df = self.extract_temporal_pattern_features(df)

        update_progress("Extracting CyberArk features...")
        df = self.extract_cyberark_features(df)

        # One-hot encode categorical features
        update_progress("One-hot encoding categorical features...")
        # Filter out categorical columns that might have all 'unknown' or very few unique values if not intended for encoding
        # Ensure categorical columns exist in the DataFrame before attempting to encode
        categorical_cols_to_encode = [col for col in self.categorical_columns if col in df.columns and df[col].nunique() > 1]

        if categorical_cols_to_encode:
            try:
                df_encoded = pd.get_dummies(df[categorical_cols_to_encode], prefix=categorical_cols_to_encode)
                # Add new encoded columns to numerical_columns. They are essentially numerical.
                self.numerical_columns.extend(df_encoded.columns.tolist())
                df = pd.concat([df.drop(columns=categorical_cols_to_encode), df_encoded], axis=1)
            except Exception as e:
                print(f"Error during one-hot encoding: {e}")
                import traceback
                traceback.print_exc()
        
        # Ensure all numerical columns are indeed numeric and fill any remaining NaNs after feature engineering
        final_numerical_columns = []
        for col in self.numerical_columns:
            if col in df.columns:
                # Attempt to convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0) # Fill remaining NaNs with 0
                final_numerical_columns.append(col)

        self.numerical_columns = final_numerical_columns # Update numerical columns list

        # Scale numerical features
        update_progress("Scaling features...")
        df = self._scale_features(df)
        
        # Update feature columns list - this should be done after all transformations
        # self.feature_columns should only contain column names that are actually in df after processing
        # and are intended to be used as features.
        
        # Collect all potential feature columns that are now numerical (either originally or one-hot encoded)
        all_current_numerical_features = list(set(self.numerical_columns + [col for col in df.columns if col.startswith('pair_')]))

        # Exclude original identifier columns, datetime columns, and any temporary columns
        exclude_cols = [
            'timestamp', 'session_start', 'session_end', 'user_id', 'action', 'resource',
            'ip_address', 'region', 'status', 'session_id', 'user_agent',
            'ip_address_str', 'session_id_filled', 'user_id_filled', # Temporary columns
            'time_diff', 'rolling_velocity', 'velocity_std', 'action_resource_pair', 'action_resource_pair_reduced', # Intermediate calculation columns
            'user_mean_hour', 'user_std_hour', '_temp_row_id', # Exclude the temporary row ID as it's not a feature
            'region_str', 'user_agent_str' # Temporary string columns for features
        ]
        # Filter out original categorical columns that are now one-hot encoded, and any duplicates
        exclude_cols.extend(categorical_cols_to_encode)

        self.feature_columns = [col for col in all_current_numerical_features if col in df.columns and col not in exclude_cols]
        
        # Remove duplicates while preserving order
        self.feature_columns = list(dict.fromkeys(self.feature_columns)) 

        # If 'is_anomaly' exists, ensure it's not treated as a feature but is kept in the DataFrame
        if 'is_anomaly' in df.columns and 'is_anomaly' in self.feature_columns:
            self.feature_columns.remove('is_anomaly')

        update_progress("Feature engineering complete.")
        
        # Restore original index before returning, if _temp_row_id was created from df.index
        # If the original df had a RangeIndex, setting it back will be simple. If it had a custom index,
        # this might need more robust handling. Assuming it starts with a default index.
        # df = df.set_index(df['_temp_row_id']).drop(columns=['_temp_row_id'])

        return df

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