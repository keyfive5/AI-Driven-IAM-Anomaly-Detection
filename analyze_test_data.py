import pandas as pd
import json
from datetime import datetime

def analyze_test_data():
    # Read the test data
    df = pd.read_csv('test_data.csv')
    
    # Convert timestamp strings to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end'] = pd.to_datetime(df['session_end'])
    
    # Find the anomalous entry
    anomaly = df[df['is_anomaly'] == True]
    
    if len(anomaly) == 0:
        print("No anomalies found in the dataset!")
        return
    
    # Get the anomalous user's normal behavior for comparison
    anomalous_user = anomaly['user_id'].iloc[0]
    user_normal = df[(df['user_id'] == anomalous_user) & (df['is_anomaly'] == False)]
    
    # Calculate some basic statistics
    print("\n=== ANOMALY ANALYSIS ===")
    print(f"\nAnomalous User: {anomalous_user}")
    print(f"Role: {anomaly['role'].iloc[0]}")
    
    # Time-based analysis
    anomaly_hour = anomaly['timestamp'].iloc[0].hour
    print(f"\nAnomaly occurred at: {anomaly['timestamp'].iloc[0]}")
    print(f"Hour of day: {anomaly_hour}")
    
    # Session duration analysis
    anomaly_duration = (anomaly['session_end'].iloc[0] - anomaly['session_start'].iloc[0]).total_seconds() / 60
    normal_duration = (user_normal['session_end'] - user_normal['session_start']).dt.total_seconds().mean() / 60
    
    print(f"\nSession Duration:")
    print(f"Anomalous session: {anomaly_duration:.1f} minutes")
    print(f"Average normal session: {normal_duration:.1f} minutes")
    
    # Action frequency analysis
    anomaly_actions = len(anomaly)
    normal_actions = len(user_normal) / len(user_normal['session_id'].unique())
    
    print(f"\nActions per session:")
    print(f"Anomalous session: {anomaly_actions} actions")
    print(f"Average normal session: {normal_actions:.1f} actions")
    
    # IP address analysis
    anomaly_ips = anomaly['ip_address'].nunique()
    normal_ips = user_normal['ip_address'].nunique() / len(user_normal['session_id'].unique())
    
    print(f"\nIP Address Changes:")
    print(f"Anomalous session: {anomaly_ips} unique IPs")
    print(f"Average normal session: {normal_ips:.1f} unique IPs")
    
    # Show the actual anomalous entries
    print("\nAnomalous Log Entries:")
    print(anomaly[['timestamp', 'action', 'ip_address', 'status', 'resource']].to_string())

if __name__ == "__main__":
    analyze_test_data() 