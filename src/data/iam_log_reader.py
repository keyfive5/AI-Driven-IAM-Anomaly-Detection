import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import uuid
import numpy as np

class IAMLogReader:
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

    def read_logs(self, file_path: str) -> pd.DataFrame:
        """Read logs from a file and convert to DataFrame."""
        raise NotImplementedError("Subclasses must implement read_logs")
    
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

    def read_aws_cloudtrail_logs(self, file_path: str) -> pd.DataFrame:
        """Reads AWS CloudTrail logs from a JSON file and flattens them into a DataFrame."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                records = data.get('Records', [])

            logs = []
            for record in records:
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

                logs.append({
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
                })
            
            df = pd.DataFrame(logs)
            return self._standardize_columns(df) # Standardize columns after initial parsing

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while reading AWS CloudTrail logs: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def read_azure_activity_logs(self, file_path: str) -> pd.DataFrame:
        """Reads Azure Activity Logs from a JSON file and flattens them into a DataFrame."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            logs = []
            for record in data:
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

                logs.append({
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
                })
            
            df = pd.DataFrame(logs)
            return self._standardize_columns(df) # Standardize columns after initial parsing

        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while reading Azure Activity Logs: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def read_synthetic_logs(self, file_path: str) -> pd.DataFrame:
        """Reads synthetic logs from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            # Synthetic data often has 'is_anomaly' column, keep it if present
            if 'is_anomaly' in df.columns:
                self.standard_columns.append('is_anomaly')
            return self._standardize_columns(df)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"An unexpected error occurred while reading synthetic logs: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def read_aws_cloudtrail_logs_from_data(self, data: dict) -> pd.DataFrame:
        """Reads AWS CloudTrail logs from a dictionary (in-memory) and flattens them into a DataFrame."""
        try:
            records = data.get('Records', [])
            logs = []
            for record in records:
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

                logs.append({
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
                })
            
            df = pd.DataFrame(logs)
            return self._standardize_columns(df) # Standardize columns after initial parsing

        except Exception as e:
            print(f"An unexpected error occurred while reading AWS CloudTrail logs from data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def read_azure_activity_logs_from_data(self, records: List[Dict]) -> pd.DataFrame:
        """Reads Azure Activity Logs from a list of dictionaries (in-memory) and flattens them into a DataFrame."""
        try:
            logs = []
            for record in records:
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

                logs.append({
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
                })
            
            df = pd.DataFrame(logs)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['session_end'] = pd.to_datetime(df['session_end'])

            return df

        except Exception as e:
            print(f"An unexpected error occurred while reading Azure Activity Logs from data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def read_logs(self, source_type: str, file_path: str = None) -> pd.DataFrame:
        """Reads logs based on the specified source type."""
        if source_type == "Synthetic Data":
            return pd.DataFrame()
        elif source_type == "AWS CloudTrail Logs":
            return self.read_aws_cloudtrail_logs(file_path)
        elif source_type == "Azure Activity Logs":
            return self.read_azure_activity_logs(file_path)
        elif source_type == "Synthetic Logs":
            return self.read_synthetic_logs(file_path)
        else:
            print(f"Error: Unknown data source type: {source_type}")
            return pd.DataFrame()

class AWSCloudTrailReader(IAMLogReader):
    """Reader for AWS CloudTrail logs."""
    
    def read_logs(self, file_path: str) -> pd.DataFrame:
        """Read AWS CloudTrail logs from a JSON file."""
        try:
            with open(file_path, 'r') as f:
                full_log_content = json.load(f) # Load the entire JSON file
            
            # Extract relevant fields
            records = []
            if 'Records' in full_log_content:
                for record in full_log_content['Records']:
                    records.append({
                        'timestamp': record.get('eventTime'),
                        'user_id': (record.get('userIdentity') or {}).get('userName'),
                        'role': (record.get('userIdentity') or {}).get('arn'),
                        'action': record.get('eventName'),
                        'ip_address': record.get('sourceIPAddress'),
                        'status': 'success' if record.get('errorCode') is None else 'failure',
                        'resource': (record.get('requestParameters') or {}).get('resourceId'),
                        'user_agent': record.get('userAgent'),
                        'region': record.get('awsRegion')
                    })
            
            return pd.DataFrame(records)
        except Exception as e:
            self.logger.error(f"Error reading AWS CloudTrail logs: {str(e)}")
            raise
    
    def validate_logs(self, df: pd.DataFrame) -> bool:
        """Validate AWS CloudTrail log structure."""
        required_columns = ['timestamp', 'user_id', 'action', 'ip_address']
        return all(col in df.columns for col in required_columns)
    
    def clean_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean AWS CloudTrail logs."""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill missing values
        df['user_id'] = df['user_id'].fillna('unknown')
        df['role'] = df['role'].fillna('unknown')
        df['ip_address'] = df['ip_address'].fillna('0.0.0.0')
        
        # Remove invalid timestamps
        df = df[df['timestamp'].notna()]
        
        return df

class AzureADReader(IAMLogReader):
    """Reader for Azure AD audit logs."""
    
    def read_logs(self, file_path: str) -> pd.DataFrame:
        """Read Azure AD audit logs from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Map Azure AD fields to our standard format
            df = df.rename(columns={
                'TimeGenerated': 'timestamp',
                'UserPrincipalName': 'user_id',
                'OperationName': 'action',
                'IPAddress': 'ip_address',
                'Result': 'status',
                'UserAgent': 'user_agent',
                'ResourceId': 'resource'
            })
            
            return df
        except Exception as e:
            self.logger.error(f"Error reading Azure AD logs: {str(e)}")
            raise
    
    def validate_logs(self, df: pd.DataFrame) -> bool:
        """Validate Azure AD log structure."""
        required_columns = ['timestamp', 'user_id', 'action', 'ip_address']
        return all(col in df.columns for col in required_columns)
    
    def clean_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Azure AD logs."""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Fill missing values
        df['user_id'] = df['user_id'].fillna('unknown')
        df['ip_address'] = df['ip_address'].fillna('0.0.0.0')
        
        # Remove invalid timestamps
        df = df[df['timestamp'].notna()]
        
        return df

class SyntheticLogReader(IAMLogReader):
    def read_logs(self, file_path: str) -> pd.DataFrame:
        return self.read_synthetic_logs(file_path)

    def validate_logs(self, df: pd.DataFrame) -> bool:
        return super().validate_logs(df)

    def clean_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        return super().clean_logs(df)

class CyberArkLogReader(IAMLogReader):
    def __init__(self):
        super().__init__()
        # Add CyberArk-specific standard columns if needed, beyond the base ones
        self.standard_columns.extend([
            'privileged_account_used',
            'vault_name',
            'session_duration_seconds',
            'is_privileged_session',
            'policy_violation',
            'reason_for_access',
            'ticket_id'
        ])

    def read_logs(self, file_path: str = None, num_events: int = 1000, anomaly_ratio: float = 0.1, num_privileged_accounts: int = 10) -> pd.DataFrame:
        """Generates synthetic CyberArk-like logs."""
        print(f"Generating {num_events} synthetic CyberArk logs...")
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

        df = pd.DataFrame(logs)
        print(f"Generated {len(df)} synthetic CyberArk logs.")
        return self._standardize_columns(df)

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
        df = reader.read_logs('path/to/cloudtrail-logs.json')
        if reader.validate_logs(df):
            df = reader.clean_logs(df)
            print(f"Successfully processed {len(df)} AWS CloudTrail logs")
            print("\nSample of processed data:")
            print(df.head())
    except Exception as e:
        print(f"Error processing AWS CloudTrail logs: {str(e)}") 