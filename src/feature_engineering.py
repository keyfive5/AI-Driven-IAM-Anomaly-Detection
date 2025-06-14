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
        # Fill NaT in timestamp before extracting features that expect valid datetime
        # Use .loc to avoid SettingWithCopyWarning
        df_temp = df.copy()
        df_temp = df_temp.dropna(subset=['timestamp'])

        df['hour'] = df_temp['timestamp'].dt.hour
        df['day_of_week'] = df_temp['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_working_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
        
        # Add cyclical time features for hour of day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'].fillna(0)/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'].fillna(0)/24)
        
        # Time since last access for each user
        df = df.sort_values(['user_id', 'timestamp'])
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
        df['session_id_filled'] = df['session_id'].fillna('unknown_session')
        df['ip_changes_in_session'] = df.groupby('session_id_filled')['ip_address_str'].transform('nunique')
        df['ip_changes_in_session'] = df['ip_changes_in_session'].fillna(0) # Fill NaN from transform with 0
        df.drop(columns=['session_id_filled'], inplace=True)

        # Add to feature columns
        self.numerical_columns.extend(['is_private_ip', 'ip_frequency', 'ip_changes_in_session'])
        self.categorical_columns.append('top_ip')
        
        # Drop the temporary column
        df.drop(columns=['ip_address_str'], inplace=True)
        
        return df
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract behavioral features based on user actions."""
        # Ensure 'user_id', 'action', 'resource', 'status' are strings and filled
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['status'] = df['status'].fillna('unknown_status').astype(str)

        # Action frequency per user
        action_counts = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
        action_counts.columns = [f'action_{col}_count' for col in action_counts.columns]
        df = df.merge(action_counts, on='user_id', how='left').fillna(0)

        # Resource access patterns
        resource_counts = df.groupby(['user_id', 'resource']).size().unstack(fill_value=0)
        resource_counts.columns = [f'resource_{col}_count' for col in resource_counts.columns]
        df = df.merge(resource_counts, on='user_id', how='left').fillna(0)

        # Success/failure ratio
        df['success_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'success').sum()
        ).fillna(0)
        df['failure_count'] = df.groupby('user_id')['status'].transform(
            lambda x: (x == 'failure').sum()
        ).fillna(0)
        df['success_ratio'] = df['success_count'] / (df['success_count'] + df['failure_count'])
        df['success_ratio'] = df['success_ratio'].fillna(1) # Fill NaN where total count is 0 (divide by zero)

        # Add rate-based features
        df_temp = df[['user_id', 'timestamp', 'action', 'resource']].copy()
        df_temp = df_temp.sort_values(by=['user_id', 'timestamp'])
        # Drop rows where timestamp is NaT before time-based calculations
        df_temp = df_temp.dropna(subset=['timestamp'])

        # Actions per minute (Manual rolling window for robustness)
        actions_per_minute_list = []
        for user_id, group in df_temp.groupby('user_id'):
            group = group.sort_values(by='timestamp').reset_index(drop=True)
            for i in range(len(group)):
                current_time = group.loc[i, 'timestamp']
                window_start = current_time - pd.Timedelta(minutes=1)
                # Select rows within the 1-minute window ending at current_time (exclusive)
                window_data = group[(group['timestamp'] >= window_start) & (group['timestamp'] < current_time)]
                actions_count = len(window_data) if not window_data.empty else 0
                actions_per_minute_list.append({
                    'user_id': user_id,
                    'timestamp': current_time,
                    'actions_per_minute': actions_count
                })
        actions_per_minute_df = pd.DataFrame(actions_per_minute_list)
        df = df.merge(actions_per_minute_df, on=['user_id', 'timestamp'], how='left').fillna({'actions_per_minute': 0})

        # Unique actions per hour (Manual rolling window for robustness)
        unique_actions_per_hour_list = []
        for user_id, group in df_temp.groupby('user_id'):
            group = group.sort_values(by='timestamp').reset_index(drop=True)
            for i in range(len(group)):
                current_time = group.loc[i, 'timestamp']
                window_start = current_time - pd.Timedelta(hours=1)
                window_data = group[(group['timestamp'] >= window_start) & (group['timestamp'] < current_time)]
                unique_count = window_data['action'].nunique() if not window_data.empty else 0
                unique_actions_per_hour_list.append({
                    'user_id': user_id,
                    'timestamp': current_time,
                    'unique_actions_per_hour': unique_count
                })
        unique_actions_per_hour_df = pd.DataFrame(unique_actions_per_hour_list)
        df = df.merge(unique_actions_per_hour_df, on=['user_id', 'timestamp'], how='left').fillna({'unique_actions_per_hour': 0})

        # Unique resources per hour (Manual rolling window for robustness)
        unique_resources_per_hour_list = []
        for user_id, group in df_temp.groupby('user_id'):
            group = group.sort_values(by='timestamp').reset_index(drop=True)
            for i in range(len(group)):
                current_time = group.loc[i, 'timestamp']
                window_start = current_time - pd.Timedelta(hours=1)
                window_data = group[(group['timestamp'] >= window_start) & (group['timestamp'] < current_time)]
                unique_count = window_data['resource'].nunique() if not window_data.empty else 0
                unique_resources_per_hour_list.append({
                    'user_id': user_id,
                    'timestamp': current_time,
                    'unique_resources_per_hour': unique_count
                })
        unique_resources_per_hour_df = pd.DataFrame(unique_resources_per_hour_list)
        df = df.merge(unique_resources_per_hour_df, on=['user_id', 'timestamp'], how='left').fillna({'unique_resources_per_hour': 0})

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
        df_temp = df.dropna(subset=['timestamp']).copy()

        # Only process sessions if there are valid session IDs beyond 'unknown_session'
        # and if there's enough data to form groups (i.e., more than one record after dropping NaNs)
        if not df_temp[df_temp['session_id'] != 'unknown_session'].empty and len(df_temp) > 0:
            # Sort within each session for time-based calculations
            df_temp = df_temp.sort_values(by=['session_id', 'timestamp'])

            # 1. Actions per session
            # Use transform to broadcast the session size back to the original DataFrame's index
            df_temp['actions_per_session'] = df_temp.groupby('session_id').transform('size')

            # 2. Distinct actions per session
            df_temp['distinct_actions_per_session'] = df_temp.groupby('session_id')['action'].transform('nunique')

            # 3. Time between actions in session (for each user within a session)
            # Calculate diff of timestamps within each session
            df_temp['time_between_actions_in_session'] = df_temp.groupby('session_id')['timestamp'].diff().dt.total_seconds().fillna(0)

            # 4. Average time between actions in session (Manual rolling mean for robustness)
            average_time_between_actions_list = []
            for session_id_val, group in df_temp.groupby('session_id'):
                group = group.sort_values(by='timestamp').reset_index(drop=True)
                # Ensure time_between_actions_in_session is available for rolling
                group['time_between_actions_in_session'] = group['timestamp'].diff().dt.total_seconds().fillna(0)
                for i in range(len(group)):
                    current_time = group.loc[i, 'timestamp']
                    window_start = current_time - pd.Timedelta(seconds=300) # 5 minutes window
                    window_data = group[(group['timestamp'] >= window_start) & (group['timestamp'] < current_time)]
                    avg_duration = window_data['time_between_actions_in_session'].mean() if not window_data.empty else 0
                    average_time_between_actions_list.append({
                        'session_id': session_id_val,
                        'timestamp': current_time,
                        'average_time_between_actions_in_session': avg_duration
                    })
            average_time_between_actions_df = pd.DataFrame(average_time_between_actions_list)

            # Merge this new dataframe back to df_temp, then df
            df_temp = df_temp.merge(average_time_between_actions_df, on=['session_id', 'timestamp'], how='left').fillna({'average_time_between_actions_in_session': 0})

            # Merge back to the original DataFrame
            # Ensure df_temp has the original index so merge works correctly
            df_temp = df_temp.set_index(df_temp.index)

            # Select only the newly created features for merging
            features_to_merge = df_temp[[
                'actions_per_session',
                'distinct_actions_per_session',
                'time_between_actions_in_session',
                'average_time_between_actions_in_session'
            ]]

            # Use left_index=True, right_index=True to merge on index
            df = df.merge(features_to_merge, left_index=True, right_index=True, how='left')

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
        top_n_regions = region_counts.head(10).index.tolist() # Consider top 10 regions
        df['top_region'] = df['region_str'].apply(lambda x: x if x in top_n_regions else 'other_region')
        
        # Region changes per session - 'session_id' and 'region' are now guaranteed to exist.
        # Ensure 'session_id' is not NA for groupby. Fill with a placeholder string if needed
        df['session_id_filled'] = df['session_id'].fillna('unknown_session')
        df['region_changes_in_session'] = df.groupby('session_id_filled')['region_str'].transform('nunique')
        df['region_changes_in_session'] = df['region_changes_in_session'].fillna(0) # Fill NaN from transform
        df.drop(columns=['session_id_filled'], inplace=True)

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
        top_n_uas = ua_counts.head(10).index.tolist() # Consider top 10 UAs
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
        df['session_id_filled'] = df['session_id'].fillna('unknown_session')
        df['user_agent_changes_in_session'] = df.groupby('session_id_filled')['user_agent_str'].transform('nunique')
        df['user_agent_changes_in_session'] = df['user_agent_changes_in_session'].fillna(0)
        df.drop(columns=['session_id_filled'], inplace=True)

        # Add to feature columns
        self.numerical_columns.extend(['user_agent_frequency', 'user_agent_changes_in_session'])
        self.categorical_columns.extend(['top_user_agent', 'os', 'browser'])

        # Drop the temporary column
        df.drop(columns=['user_agent_str'], inplace=True)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
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
        if not hasattr(self.scaler, 'mean_'):
            # Filter df to only numerical columns that are not all zeros or NaNs
            # to avoid issues with fitting on constant features
            train_data = df[cols_to_scale].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
            
            if not train_data.empty:
                self.scaler.fit(train_data)
            else:
                print("Warning: No varying numerical data to fit scaler. Skipping scaling for this batch.")
                return df

        # Transform data. Handle potential new columns during transform by re-filtering.
        scaled_data = self.scaler.transform(df[cols_to_scale])
        df[cols_to_scale] = scaled_data
        
        print(f"Scaled {len(cols_to_scale)} numerical features.")
        return df

    def engineer_features(self, df: pd.DataFrame, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> pd.DataFrame:
        """Main method to engineer all features."""
        self.feature_columns = [] # Reset for each run
        self.categorical_columns = []
        self.numerical_columns = []

        total_steps = 8 # Adjusted for scaling step
        current_step = 0

        def update_progress(message):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(current_step, total_steps, message)

        update_progress("Starting feature engineering...")
        
        # Add a unique identifier for each row before any merges/resets
        # This ensures we can always merge back to the original rows if needed.
        df['_row_id'] = df.index # Use original DataFrame index as a stable row ID

        # Convert key columns to string types early and fill NAs to prevent errors in grouping/merging
        df['user_id'] = df['user_id'].fillna('unknown_user').astype(str)
        df['action'] = df['action'].fillna('unknown_action').astype(str)
        df['resource'] = df['resource'].fillna('unknown_resource').astype(str)
        df['status'] = df['status'].fillna('unknown_status').astype(str)
        df['ip_address'] = df['ip_address'].fillna('unknown_ip').astype(str)
        df['region'] = df['region'].fillna('unknown_region').astype(str)
        df['user_agent'] = df['user_agent'].fillna('unknown_user_agent').astype(str)
        df['session_id'] = df['session_id'].fillna('unknown_session').astype(str)

        update_progress("Extracting time-based features...")
        df = self.extract_time_features(df)

        update_progress("Extracting IP-based features...")
        df = self.extract_ip_features(df)

        update_progress("Extracting behavioral features...")
        df = self.extract_behavioral_features(df)

        update_progress("Extracting session-based features...")
        df = self.extract_session_features(df)

        update_progress("Extracting region-based features...")
        df = self.extract_region_features(df)

        update_progress("Extracting user agent features...")
        df = self.extract_user_agent_features(df)

        # One-hot encode categorical features
        update_progress("One-hot encoding categorical features...")
        # Filter out categorical columns that might have all 'unknown' or very few unique values if not intended for encoding
        # Ensure categorical columns exist in the DataFrame before attempting to encode
        categorical_cols_to_encode = [col for col in self.categorical_columns if col in df.columns and df[col].nunique() > 1]

        if categorical_cols_to_encode:
            try:
                df_encoded = pd.get_dummies(df[categorical_cols_to_encode], prefix=categorical_cols_to_encode)
                # Align columns - crucial for consistent model input
                for col in df_encoded.columns:
                    if col not in self.feature_columns:
                        self.feature_columns.append(col)
                
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

        # Apply feature scaling after all features are engineered and numerical columns are finalized
        update_progress("Scaling numerical features...")
        df = self._scale_features(df)

        # Final feature column list should include numerical and encoded categorical
        self.feature_columns = list(set(self.numerical_columns + [col for col in df.columns if '_' in col and any(cat_col in col for cat_col in categorical_cols_to_encode)]))

        # Exclude original identifier columns, datetime columns, and any temporary columns
        # This list should be managed carefully to ensure no features are accidentally dropped
        exclude_cols = [
            'timestamp', 'session_start', 'session_end', 'user_id', 'action', 'resource',
            'ip_address', 'region', 'status', 'session_id', 'user_agent', '_row_id',
            'is_anomaly' # Keep if for training, exclude for prediction
        ]
        # Filter out original categorical columns that are now one-hot encoded
        exclude_cols.extend(categorical_cols_to_encode)

        # Filter self.feature_columns to only include those actually in df and not in exclude_cols
        self.feature_columns = [col for col in self.feature_columns if col in df.columns and col not in exclude_cols]
        
        # Ensure no duplicates in feature_columns
        self.feature_columns = list(dict.fromkeys(self.feature_columns)) # Preserve order while removing duplicates

        # If 'is_anomaly' exists, keep it in the dataframe but exclude from features
        if 'is_anomaly' in df.columns and 'is_anomaly' not in self.feature_columns:
            # Make sure it's at the end or handled separately
            pass # It will be handled outside by main.py for training/evaluation

        update_progress("Feature engineering complete.")
        print(f"DEBUG: Final features to return: {self.feature_columns[:5]}...") # Debug print
        
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