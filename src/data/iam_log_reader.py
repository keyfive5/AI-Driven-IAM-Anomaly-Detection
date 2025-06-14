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