import pandas as pd
import json
from typing import Dict, List, Optional
from datetime import datetime
import logging

class IAMLogReader:
    """Base class for reading IAM logs from different sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def read_logs(self, file_path: str) -> pd.DataFrame:
        """Read logs from a file and convert to DataFrame."""
        raise NotImplementedError("Subclasses must implement read_logs")
    
    def validate_logs(self, df: pd.DataFrame) -> bool:
        """Validate the structure and content of the logs."""
        raise NotImplementedError("Subclasses must implement validate_logs")
    
    def clean_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the logs."""
        raise NotImplementedError("Subclasses must implement clean_logs")

    def read_aws_cloudtrail_logs(self, file_path: str) -> pd.DataFrame:
        """Reads AWS CloudTrail logs from a JSON file and flattens them into a DataFrame."""
        try:
            with open(file_path, 'r') as f:
                # CloudTrail logs are often a list of records under a 'Records' key
                data = json.load(f)
                records = data.get('Records', [])

            logs = []
            for record in records:
                event_time = record.get('eventTime')
                # Handle different identity types
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
                event_id = record.get('eventID')

                # Determine status based on errorCode or errorMessage presence
                status = 'failure' if error_code or error_message else 'success'

                logs.append({
                    'timestamp': event_time,
                    'user_id': user_id,
                    'action': event_name,
                    'resource': resource_name,
                    'ip_address': ip_address,
                    'region': aws_region,
                    'status': status,
                    'session_id': request_id, # Using requestID as session_id for CloudTrail
                    'session_start': event_time, # For simplicity, start and end are eventTime
                    'session_end': event_time,
                    'user_agent': user_agent
                })
            
            df = pd.DataFrame(logs)
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['session_end'] = pd.to_datetime(df['session_end'])
            
            print(f"Successfully loaded {len(df)} AWS CloudTrail logs.")
            return df

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
                
                # Extract user_id. Azure uses 'oid' (object ID) or 'name' in claims
                identity = record.get('identity', {}).get('claims', {})
                user_id = identity.get('oid') or identity.get('name')
                
                operation_name = record.get('operationName')
                resource_id = record.get('resourceId')
                caller_ip_address = record.get('callerIpAddress')
                correlation_id = record.get('correlationId') # Can serve as a session ID
                result_type = record.get('resultType') # 'Success', 'Failed', etc.

                # Extract resource type/name from resourceId if possible
                resource_parts = resource_id.split('/') if resource_id else []
                resource = resource_parts[-1] if len(resource_parts) > 0 else None
                # A more generic action could be the last part of operationName
                action = operation_name.split('/')[-1] if operation_name else None

                # Map Azure resultType to a simplified status
                status = 'success' if result_type and result_type.lower() == 'success' else 'failure'

                logs.append({
                    'timestamp': event_time,
                    'user_id': user_id,
                    'action': action,
                    'resource': resource,
                    'ip_address': caller_ip_address,
                    'region': None, # Azure logs don't directly provide 'region' in this format, set to None
                    'status': status,
                    'session_id': correlation_id, # Using correlationId as session_id
                    'session_start': event_time,
                    'session_end': event_time,
                    'user_agent': None # Azure logs don't directly provide 'user_agent' in this format, set to None
                })
            
            df = pd.DataFrame(logs)
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['session_end'] = pd.to_datetime(df['session_end'])

            print(f"Successfully loaded {len(df)} Azure Activity Logs.")
            return df

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
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['session_end'] = pd.to_datetime(df['session_end'])
            
            return df

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

def get_log_reader(source: str) -> IAMLogReader:
    """Factory function to get the appropriate log reader."""
    readers = {
        'aws': AWSCloudTrailReader,
        'azure': AzureADReader
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