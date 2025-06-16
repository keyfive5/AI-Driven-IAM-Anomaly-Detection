import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from utils.logging_config import get_logger

class FeatureEngineer:
    def __init__(self):
        self.logger = get_logger('analysis')
        self.feature_columns = []
        
    def engineer_features(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
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
            progress_callback(0, "Extracting time-based features...")
        self.logger.debug("Extracting time-based features")
        df_features = self._extract_time_features(df_features)
        
        # 2. User behavior features
        if progress_callback:
            progress_callback(20, "Extracting user behavior features...")
        self.logger.debug("Extracting user behavior features")
        df_features = self._extract_user_behavior_features(df_features)
        
        # 3. Session-based features
        if progress_callback:
            progress_callback(40, "Extracting session-based features...")
        self.logger.debug("Extracting session features")
        df_features = self._extract_session_features(df_features)
        
        # 4. Resource access patterns
        if progress_callback:
            progress_callback(60, "Extracting resource access patterns...")
        self.logger.debug("Extracting resource access features")
        df_features = self._extract_resource_access_features(df_features)
        
        # 5. IP and location features
        if progress_callback:
            progress_callback(80, "Extracting IP and location features...")
        self.logger.debug("Extracting IP and location features")
        df_features = self._extract_ip_location_features(df_features)
        
        # Store the feature columns
        self.feature_columns = [col for col in df_features.columns if col not in original_columns]
        self.logger.info(f"Generated {len(self.feature_columns)} features")
        self.logger.debug(f"Feature columns: {self.feature_columns}")
        
        if progress_callback:
            progress_callback(100, "Feature engineering complete!")
            
        return df_features
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from the logs."""
        self.logger.debug("Extracting time-based features")
        try:
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Extract time components
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_working_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            # Time since last action for each user
            df['time_since_last_action'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
            
            # Rolling time windows
            df['actions_last_hour'] = df.groupby('user_id')['timestamp'].transform(
                lambda x: x.rolling('1H', min_periods=1).count()
            )
            
            self.logger.debug("Time-based features extracted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting time features: {e}", exc_info=True)
            raise
    
    def _extract_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract user behavior features."""
        self.logger.debug("Extracting user behavior features")
        try:
            # Action frequency per user
            action_counts = df.groupby(['user_id', 'action']).size().unstack(fill_value=0)
            action_counts.columns = [f'action_count_{col}' for col in action_counts.columns]
            df = df.merge(action_counts, on='user_id', how='left')
            
            # Success/failure ratio
            df['success_count'] = df.groupby('user_id')['status'].transform(
                lambda x: (x == 'success').sum()
            )
            df['failure_count'] = df.groupby('user_id')['status'].transform(
                lambda x: (x == 'failure').sum()
            )
            df['success_ratio'] = df['success_count'] / (df['success_count'] + df['failure_count'])
            
            # Unique resources accessed
            df['unique_resources'] = df.groupby('user_id')['resource'].transform('nunique')
            
            self.logger.debug("User behavior features extracted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting user behavior features: {e}", exc_info=True)
            raise
    
    def _extract_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract session-based features."""
        self.logger.debug("Extracting session features")
        try:
            # Session duration
            df['session_duration'] = (df['session_end'] - df['session_start']).dt.total_seconds()
            
            # Actions per session
            df['actions_per_session'] = df.groupby('session_id').transform('size')
            
            # Session frequency
            df['sessions_per_user'] = df.groupby('user_id')['session_id'].transform('nunique')
            
            self.logger.debug("Session features extracted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting session features: {e}", exc_info=True)
            raise
    
    def _extract_resource_access_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract resource access pattern features."""
        self.logger.debug("Extracting resource access features")
        try:
            # Resource access frequency
            df['resource_access_count'] = df.groupby(['user_id', 'resource']).transform('size')
            
            # Resource access time patterns
            df['resource_access_hour'] = df.groupby(['user_id', 'resource'])['hour'].transform('mean')
            df['resource_access_std'] = df.groupby(['user_id', 'resource'])['hour'].transform('std')
            
            self.logger.debug("Resource access features extracted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting resource access features: {e}", exc_info=True)
            raise
    
    def _extract_ip_location_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract IP and location-based features."""
        self.logger.debug("Extracting IP and location features")
        try:
            # IP address frequency
            df['ip_frequency'] = df.groupby('ip_address').transform('size')
            
            # IP address changes per user
            df['ip_changes'] = df.groupby('user_id')['ip_address'].transform(
                lambda x: x.ne(x.shift()).cumsum()
            )
            
            # Region changes
            df['region_changes'] = df.groupby('user_id')['region'].transform(
                lambda x: x.ne(x.shift()).cumsum()
            )
            
            self.logger.debug("IP and location features extracted successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error extracting IP and location features: {e}", exc_info=True)
            raise
    
    def get_feature_columns(self) -> List[str]:
        """Get the list of engineered feature columns."""
        return self.feature_columns 