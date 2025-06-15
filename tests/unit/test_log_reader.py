import unittest
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from src.data.iam_log_reader import IAMLogReader, get_log_reader, AWSCloudTrailReader, AzureADReader

class TestIAMLogReader(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.reader = AWSCloudTrailReader()
        self.sample_aws_log = {
            "eventVersion": "1.08",
            "userIdentity": {"type": "IAMUser", "principalId": "AIDAISAEXAMPLE", "arn": "arn:aws:iam::123456789012:user/testuser", "accountId": "123456789012", "userName": "testuser"},
            "eventTime": "2023-10-26T10:00:00Z",
            "eventSource": "s3.amazonaws.com",
            "eventName": "GetObject",
            "awsRegion": "us-east-1",
            "sourceIPAddress": "192.168.1.1",
            "userAgent": "Mozilla/5.0",
            "requestParameters": {"bucketName": "my-example-bucket", "key": "data/file.txt"},
            "responseElements": None,
            "requestID": "ABCD12345",
            "eventID": "EFGH67890",
            "readOnly": True,
            "resources": [{
                "accountId": "123456789012",
                "ARN": "arn:aws:s3:::my-example-bucket"
            }],
            "eventType": "AwsApiCall",
            "recipientAccountId": "123456789012",
            "sessionContext": {"sessionIssuer": {"type": "Role", "principalId": "AROAISAEXAMPLE", "arn": "arn:aws:iam::123456789012:role/testrole", "accountId": "123456789012", "userName": "testrole"}, "webIdFederationData": None, "attributes": {"mfaAuthenticated": "false", "creationDate": "2023-10-26T09:00:00Z"}},
            "eventCategory": "Management",
            "managementEvent": True,
            "eventTypeCode": "AwsApiCall",
            "apiVersion": "1.1"
        }

        self.sample_azure_log = {
            "time": "2023-10-26T10:00:00Z",
            "resourceId": "/subscriptions/testsub/resourceGroups/testrg/providers/Microsoft.Compute/virtualMachines/testvm",
            "category": "Administrative",
            "operationName": "Microsoft.Compute/virtualMachines/start/action",
            "resourceGroup": "testrg",
            "resourceProvider": "Microsoft.Compute",
            "status": {"value": "Succeeded"},
            "subStatus": "",
            "caller": "testuser@example.com",
            "correlationId": "AZURECORRELATION123",
            "eventTimestamp": "2023-10-26T10:00:00Z",
            "httpRequest": {"clientIpAddress": "192.168.1.2"},
            "properties": {"message": "Virtual machine started"}
        }

        # Create dummy log files for testing file reading
        self.aws_log_file = "temp_aws_logs.json"
        self.azure_log_file = "temp_azure_logs.json"
        with open(self.aws_log_file, 'w') as f:
            json.dump(self.sample_aws_log, f)
        with open(self.azure_log_file, 'w') as f:
            json.dump(self.sample_azure_log, f)

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.aws_log_file):
            os.remove(self.aws_log_file)
        if os.path.exists(self.azure_log_file):
            os.remove(self.azure_log_file)
        
    def test_standard_columns(self):
        """Test that standard columns are properly defined."""
        required_columns = [
            'timestamp', 'user_id', 'action', 'resource', 'ip_address',
            'region', 'status', 'session_id', 'session_start', 'session_end',
            'user_agent'
        ]
        self.assertEqual(self.reader.standard_columns, required_columns)
        
    def test_column_standardization(self):
        """Test column standardization functionality."""
        # Create a dummy DataFrame with some missing/misnamed columns
        data = {
            'Time': ['2023-01-01T10:00:00Z'],
            'User': ['user1'],
            'ActionType': ['login'],
            'SourceIP': ['1.1.1.1']
        }
        df = pd.DataFrame(data)
        standardized_df = self.reader.standardize_columns(df)
        
        self.assertTrue(all(col in standardized_df.columns for col in self.reader.standard_columns))
        self.assertTrue(standardized_df['resource'].isna().all()) # resource should be NaN
        self.assertTrue(standardized_df['session_id'].isna().all()) # session_id should be NaN
        
    def test_timestamp_conversion(self):
        """Test timestamp conversion in standardization."""
        data = {
            'Time': ['2023-01-01T10:00:00Z'],
            'User': ['user1'],
            'ActionType': ['login'],
            'SourceIP': ['1.1.1.1']
        }
        df = pd.DataFrame(data)
        standardized_df = self.reader.standardize_columns(df)
        
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(standardized_df['timestamp']))
        self.assertEqual(standardized_df['timestamp'].iloc[0], datetime(2023, 1, 1, 10, 0, 0))
        
    def test_log_validation(self):
        """Test log validation functionality."""
        # Valid DataFrame
        valid_df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'user_id': ['user1'],
            'action': ['action1'],
            'resource': ['resource1'],
            'ip_address': ['1.1.1.1'],
            'region': ['region1'],
            'status': ['success'],
            'session_id': ['session1'],
            'session_start': [datetime.now() - timedelta(minutes=10)],
            'session_end': [datetime.now()],
            'user_agent': ['agent1']
        })
        self.assertTrue(self.reader.validate_logs(valid_df))
        
        # Empty DataFrame
        empty_df = pd.DataFrame(columns=self.reader.standard_columns)
        self.assertFalse(self.reader.validate_logs(empty_df))
        
        # Missing required column
        invalid_df = valid_df.drop(columns=['user_id'])
        self.assertFalse(self.reader.validate_logs(invalid_df))
        
        # Incorrect data type for timestamp should pass initial validation if column exists, then fail later processing
        # The `validate_logs` method checks for column presence and all-null, not type correctness per se.
        invalid_df_dtype = valid_df.copy()
        invalid_df_dtype['timestamp'] = ['not-a-timestamp']
        self.assertTrue(self.reader.validate_logs(invalid_df_dtype)) # This should now pass as column exists and is not all null

    def test_aws_log_reader(self):
        """Test AWS CloudTrail log reader functionality."""
        reader = AWSCloudTrailReader()
        df_chunks = list(reader.read_logs_in_chunks(self.aws_log_file)) # Collect all chunks
        df = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()
        self.assertFalse(df.empty)
        self.assertTrue(all(col in df.columns for col in reader.standard_columns))
        self.assertEqual(df['user_id'].iloc[0], 'testuser')
        self.assertEqual(df['action'].iloc[0], 'GetObject')
        self.assertEqual(df['ip_address'].iloc[0], '192.168.1.1')
        self.assertEqual(df['region'].iloc[0], 'us-east-1')
        self.assertEqual(df['status'].iloc[0], 'success')
        self.assertIsNotNone(df['session_id'].iloc[0])
        
    def test_azure_log_reader(self):
        """Test Azure Activity log reader functionality."""
        reader = AzureADReader()
        df_chunks = list(reader.read_logs_in_chunks(self.azure_log_file)) # Collect all chunks
        df = pd.concat(df_chunks, ignore_index=True) if df_chunks else pd.DataFrame()
        self.assertFalse(df.empty)
        self.assertTrue(all(col in df.columns for col in reader.standard_columns))
        self.assertEqual(df['user_id'].iloc[0], 'testuser@example.com')
        self.assertEqual(df['action'].iloc[0], 'Microsoft.Compute/virtualMachines/start/action')
        self.assertEqual(df['ip_address'].iloc[0], '192.168.1.2')
        self.assertIsNone(df['region'].iloc[0]) # Azure logs may not have a direct region field, often inferred from resourceId
        self.assertEqual(df['status'].iloc[0], 'Succeeded')
        self.assertIsNotNone(df['session_id'].iloc[0])

if __name__ == '__main__':
    unittest.main() 