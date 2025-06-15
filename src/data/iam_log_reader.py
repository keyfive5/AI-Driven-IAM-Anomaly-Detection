from abc import ABC, abstractmethod
import pandas as pd
import json
from typing import Dict, List, Optional, Generator
from datetime import datetime, timedelta
import logging
import uuid
import numpy as np

class IAMLogReader(ABC):
    """Base class for reading IAM logs from different sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.standard_columns = [
            'timestamp', 'user_id', 'action', 'resource', 'ip_address',
            'region', 'status', 'session_id', 'session_start', 'session_end',
            'user_agent'
        ]

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes column names and ensures all expected columns exist.
        Missing columns will be added with NaN values.
        """
        df = df.copy() # Work on a copy to avoid modifying the original DataFrame passed in
        for col in self.standard_columns:
            if col not in df.columns:
                df[col] = pd.NA # Use pd.NA for nullable columns

        # Ensure timestamp columns are datetime objects
        for col in ['timestamp', 'session_start', 'session_end']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df[self.standard_columns] # Return DataFrame with standard columns in defined order

    @abstractmethod
    def _parse_single_log_record(self, record: Dict) -> Dict:
        """Abstract method to parse a single raw log record into a standardized format."""
        pass

    def read_logs_in_chunks(self, file_path: str, chunk_size: int = 10000) -> Generator[pd.DataFrame, None, None]:
        """Reads logs from a file in chunks and yields DataFrames."""
        try:
            with open(file_path, 'r') as f:
                # For JSON files where the entire content is a list of records (e.g., Azure Activity Logs)
                # or a dict containing a list (e.g., AWS CloudTrail), we need to load it all first.
                # This means chunking is done in memory after initial load.
                # For very large files, a different approach (e.g., iterating lines) would be needed.
                full_data = json.load(f)

            records = []
            if isinstance(full_data, dict) and 'Records' in full_data: # AWS CloudTrail format
                records = full_data['Records']
            elif isinstance(full_data, list): # Azure Activity Log format
                records = full_data
            else:
                self.logger.error(f"Unsupported log file format: {file_path}")
                yield pd.DataFrame() # Yield an empty DataFrame
                return

            self.logger.info(f"Total records to process: {len(records)}")
            
            chunk_data = []
            for i, record in enumerate(records):
                try:
                    standardized_record = self._parse_single_log_record(record)
                    chunk_data.append(standardized_record)
                except Exception as e:
                    self.logger.warning(f"Failed to parse record {i+1}: {e}")
                    continue # Skip problematic record

                if (i + 1) % chunk_size == 0 or (i + 1) == len(records):
                    df = pd.DataFrame(chunk_data)
                    df = self._standardize_columns(df) # Standardize and ensure required columns
                    df = self.clean_logs(df) # Apply generic cleaning
                    yield df
                    chunk_data = [] # Reset for next chunk

        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            yield pd.DataFrame()
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from {file_path}: {e}")
            yield pd.DataFrame()
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during log reading in chunks: {e}")
            self.logger.exception("Full traceback:") # Log full traceback
            yield pd.DataFrame()
            
    def validate_logs(self, df: pd.DataFrame) -> bool:
        """Validate the structure and content of the logs."""
        # Basic validation: check if essential columns exist and have non-null values
        if df.empty:
            print("Warning: Empty DataFrame after standardization.")
            return False
        
        required_columns = ['timestamp', 'user_id', 'action', 'ip_address', 'status']
        for col in required_columns:
            if col not in df.columns or df[col].isnull().all():
                print(f"Validation Error: Required column '{col}' is missing or all null after standardization.")
                return False
        
        # Further checks could be added here (e.g., timestamp format, IP address validity)
        
        return True
    
    def clean_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the logs. This will be more generic now."""
        # Fill missing string/object columns with 'unknown'
        for col in ['user_id', 'action', 'resource', 'ip_address', 'region', 'status', 'user_agent']:
            if col in df.columns:
                df[col] = df[col].fillna('unknown')
        
        # Fill missing session IDs if they are NA after standardization
        if 'session_id' in df.columns:
            # Generate a unique session ID for each row that has a missing session_id
            missing_session_mask = df['session_id'].isna()
            if missing_session_mask.any():
                df.loc[missing_session_mask, 'session_id'] = df.loc[missing_session_mask].apply(
                    lambda row: f"generated_session_{row.name}_{uuid.uuid4().hex[:8]}", axis=1
                )

        # Fill missing session_start/end with timestamp if session_id is present but start/end are missing
        if 'session_id' in df.columns and 'timestamp' in df.columns:
            # Vectorized assignment for session_start
            mask_start = df['session_start'].isna() & df['session_id'].notna()
            df.loc[mask_start, 'session_start'] = df.loc[mask_start, 'timestamp']

            # Vectorized assignment for session_end
            mask_end = df['session_end'].isna() & df['session_id'].notna()
            df.loc[mask_end, 'session_end'] = df.loc[mask_end, 'timestamp']
            
        return df

class AWSCloudTrailReader(IAMLogReader):
    def _parse_single_log_record(self, record: Dict) -> Dict:
        event_time = record.get('eventTime')
        user_identity = record.get('userIdentity', {})
        user_id = user_identity.get('userName') or user_identity.get('principalId') or user_identity.get('sessionContext', {}).get('sessionIssuer', {}).get('userName') or user_identity.get('arn')
        
        event_name = record.get('eventName')
        event_source = record.get('eventSource')
        resource_name = record.get('resources',[{}])[0].get('resourceName') if record.get('resources') else None
        ip_address = record.get('sourceIPAddress')
        aws_region = record.get('awsRegion')
        error_code = record.get('errorCode')
        error_message = record.get('errorMessage')
        user_agent = record.get('userAgent')
        request_id = record.get('requestID')

        status = 'failure' if error_code or error_message else 'success'

        return {
            'timestamp': event_time,
            'user_id': user_id,
            'action': event_name,
            'resource': resource_name,
            'ip_address': ip_address,
            'region': aws_region,
            'status': status,
            'session_id': request_id, 
            'session_start': event_time, 
            'session_end': event_time,
            'user_agent': user_agent
        }

class AzureADReader(IAMLogReader):
    def _parse_single_log_record(self, record: Dict) -> Dict:
        event_time = record.get('time')
        
        identity = record.get('identity', {}).get('claims', {})
        user_id = identity.get('oid') or identity.get('name')
        
        operation_name = record.get('operationName')
        resource_id = record.get('resourceId')
        caller_ip_address = record.get('callerIpAddress')
        correlation_id = record.get('correlationId')
        result_type = record.get('resultType')

        resource_parts = resource_id.split('/') if resource_id else []
        resource = resource_parts[-1] if len(resource_parts) > 0 else None
        action = operation_name.split('/')[-1] if operation_name else None

        status = 'success' if result_type and result_type.lower() == 'success' else 'failure'

        return {
            'timestamp': event_time,
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'ip_address': caller_ip_address,
            'region': None, 
            'status': status,
            'session_id': correlation_id, 
            'session_start': event_time,
            'session_end': event_time,
            'user_agent': None 
        }

class SyntheticLogReader(IAMLogReader):
    def _parse_single_log_record(self, record: Dict) -> Dict:
        # For synthetic logs, the record is already in a standardized dictionary format
        # We just return it as is, or with minimal processing
        return record

    def read_logs_in_chunks(self, file_path: str = None, num_events: int = 1000, anomaly_ratio: float = 0.1, num_privileged_accounts: int = 10) -> Generator[pd.DataFrame, None, None]:
        """Generates synthetic logs in chunks."""
        self.logger.info(f"Generating {num_events} synthetic logs...")
        # Assuming IAMLogGenerator is available globally or imported - will need to be passed or instantiated
        from data_generator import IAMLogGenerator
        generator = IAMLogGenerator() 
        df_full = generator.generate_dataset(n_events=num_events, anomaly_ratio=anomaly_ratio)
        
        # Yield in chunks
        chunk_size = num_events // 10 if num_events // 10 > 0 else 1 # Ensure chunk size is at least 1
        for i in range(0, len(df_full), chunk_size):
            chunk_df = df_full.iloc[i:i + chunk_size].copy()
            yield self._standardize_columns(chunk_df) # Standardize and ensure required columns

class CyberArkLogReader(IAMLogReader):
    def __init__(self):
        super().__init__()
        self.standard_columns.extend([
            'privileged_account_used',
            'vault_name',
            'session_duration_seconds',
            'is_privileged_session',
            'policy_violation',
            'reason_for_access',
            'ticket_id'
        ])
    
    def _parse_single_log_record(self, record: Dict) -> Dict:
        # For CyberArk synthetic logs, the record is already in a standardized dictionary format
        return record

    def read_logs_in_chunks(self, file_path: str = None, num_events: int = 1000, anomaly_ratio: float = 0.1, num_privileged_accounts: int = 10) -> Generator[pd.DataFrame, None, None]:
        """Generates synthetic CyberArk-like logs in chunks."""
        self.logger.info(f"Generating {num_events} synthetic CyberArk logs in chunks...")
        logs = []
        privileged_accounts = [f'privileged_account_{i}' for i in range(num_privileged_accounts)]
        regular_users = [f'user_{i}' for i in range(50)]
        vaults = [f'Vault_{i}' for i in range(5)]
        target_resources = [f'Server_{i}' for i in range(20)]
        action_types = ['Logon', 'RetrievePassword', 'Connect', 'ChangePassword', 'View', 'RotatePassword']
        ip_addresses = [f'192.168.1.{i}' for i in range(50)] + [f'10.0.0.{i}' for i in range(20)] # Mix of private and public-like
        
        start_time = datetime.now() - timedelta(days=7)

        for i in range(num_events):
            event_time = start_time + timedelta(seconds=i * 60 * np.random.rand() * 5) # Events spread over time
            
            is_anomaly = np.random.rand() < anomaly_ratio

            user_id = np.random.choice(regular_users)
            privileged_account_used = np.random.choice(privileged_accounts)
            action_type = np.random.choice(action_types)
            target_resource = np.random.choice(target_resources)
            vault_name = np.random.choice(vaults)
            ip_address = np.random.choice(ip_addresses)
            session_id = str(uuid.uuid4())
            status = 'success' if np.random.rand() > 0.1 else 'failure' # 10% failure rate
            session_duration_seconds = int(np.random.normal(300, 100)) # Avg 5 min session, std 100 sec
            if session_duration_seconds < 10: session_duration_seconds = 10
            is_privileged_session = True # For now, assume all generated are privileged for focus
            policy_violation = False
            reason_for_access = 'Routine access'
            ticket_id = None

            if is_anomaly:
                anomaly_type = np.random.choice([
                    'unusual_time', 'unusual_action', 'unusual_ip', 
                    'excessive_duration', 'policy_violation'
                ])

                if anomaly_type == 'unusual_time':
                    event_time = start_time + timedelta(days=np.random.randint(7, 14)) # Far future/past
                    event_time += timedelta(hours=int(np.random.choice([0, 23]))) # Midnight or late night
                elif anomaly_type == 'unusual_action':
                    action_type = 'UnauthorizedFileAccess' # A fabricated anomalous action
                    status = 'failure'
                elif anomaly_type == 'unusual_ip':
                    ip_address = f'203.0.113.{np.random.randint(1,254)}' # Public IP not typical
                elif anomaly_type == 'excessive_duration':
                    session_duration_seconds = int(np.random.normal(3600, 1800)) # Very long session (avg 1 hour)
                elif anomaly_type == 'policy_violation':
                    policy_violation = True
                    reason_for_access = 'No valid reason provided'
                    status = 'failure'

            logs.append({
                'timestamp': event_time,
                'user_id': user_id,
                'privileged_account_used': privileged_account_used,
                'action': action_type,
                'resource': target_resource,
                'ip_address': ip_address,
                'region': 'unknown_region', # CyberArk logs might not always have direct region
                'status': status,
                'session_id': session_id,
                'session_start': event_time, 
                'session_end': event_time + timedelta(seconds=session_duration_seconds),
                'user_agent': 'CyberArk-Client',
                'vault_name': vault_name,
                'session_duration_seconds': session_duration_seconds,
                'is_privileged_session': is_privileged_session,
                'policy_violation': policy_violation,
                'reason_for_access': reason_for_access,
                'ticket_id': ticket_id,
                'is_anomaly': is_anomaly # Label for supervised learning/evaluation
            })

        df_full = pd.DataFrame(logs)
        self.logger.info(f"Generated {len(df_full)} synthetic CyberArk logs.")

        # Yield in chunks
        chunk_size = num_events // 10 if num_events // 10 > 0 else 1 # Ensure chunk size is at least 1
        for i in range(0, len(df_full), chunk_size):
            chunk_df = df_full.iloc[i:i + chunk_size].copy()
            yield self._standardize_columns(chunk_df) # Standardize and ensure required columns

def get_log_reader(source: str) -> IAMLogReader:
    """Factory function to get the appropriate log reader."""
    readers = {
        'aws': AWSCloudTrailReader,
        'azure': AzureADReader,
        'synthetic': SyntheticLogReader,
        'cyberark': CyberArkLogReader
    }
    
    if source.lower() not in readers:
        raise ValueError(f"Unsupported log source: {source}")
    
    return readers[source.lower()]()

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with AWS CloudTrail logs
    try:
        reader = get_log_reader('aws')
        df = reader.read_logs_in_chunks('path/to/cloudtrail-logs.json')
        if reader.validate_logs(df):
            df = reader.clean_logs(df)
            print(f"Successfully processed {len(df)} AWS CloudTrail logs")
            print("\nSample of processed data:")
            print(df.head())
    except Exception as e:
        print(f"Error processing AWS CloudTrail logs: {str(e)}") 