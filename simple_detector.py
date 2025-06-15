import pandas as pd
from datetime import datetime, time

class SimpleAnomalyDetector:
    def __init__(self):
        # Define thresholds based on our analysis
        self.working_hours = (time(9, 0), time(17, 0))  # 9 AM to 5 PM
        self.max_session_duration = 180  # minutes
        self.max_actions_per_session = 20
        self.max_ip_changes = 5
        self.max_failure_rate = 0.2  # 20% failure rate threshold

    def detect_anomalies(self, df):
        # Convert timestamp strings to datetime if needed
        if isinstance(df['timestamp'].iloc[0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['session_start'] = pd.to_datetime(df['session_start'])
            df['session_end'] = pd.to_datetime(df['session_end'])

        # Group by session to analyze each session
        sessions = df.groupby('session_id')
        anomalies = []

        for session_id, session_data in sessions:
            # Calculate session metrics
            session_duration = (session_data['session_end'].iloc[0] - session_data['session_start'].iloc[0]).total_seconds() / 60
            n_actions = len(session_data)
            n_unique_ips = session_data['ip_address'].nunique()
            failure_rate = (session_data['status'] == 'failure').mean()
            start_time = session_data['timestamp'].iloc[0].time()

            # Check for anomalies
            is_anomaly = False
            reasons = []

            # Check time of day
            if not (self.working_hours[0] <= start_time <= self.working_hours[1]):
                is_anomaly = True
                reasons.append(f"Session started outside working hours at {start_time}")

            # Check session duration
            if session_duration > self.max_session_duration:
                is_anomaly = True
                reasons.append(f"Session duration {session_duration:.1f} minutes exceeds threshold of {self.max_session_duration} minutes")

            # Check number of actions
            if n_actions > self.max_actions_per_session:
                is_anomaly = True
                reasons.append(f"Number of actions ({n_actions}) exceeds threshold of {self.max_actions_per_session}")

            # Check IP changes
            if n_unique_ips > self.max_ip_changes:
                is_anomaly = True
                reasons.append(f"Number of unique IPs ({n_unique_ips}) exceeds threshold of {self.max_ip_changes}")

            # Check failure rate
            if failure_rate > self.max_failure_rate:
                is_anomaly = True
                reasons.append(f"Failure rate {failure_rate:.1%} exceeds threshold of {self.max_failure_rate:.1%}")

            if is_anomaly:
                anomalies.append({
                    'session_id': session_id,
                    'user_id': session_data['user_id'].iloc[0],
                    'start_time': session_data['session_start'].iloc[0],
                    'duration': session_duration,
                    'n_actions': n_actions,
                    'n_unique_ips': n_unique_ips,
                    'failure_rate': failure_rate,
                    'reasons': reasons
                })

        return anomalies

def main():
    # Read the test data
    df = pd.read_csv('test_data.csv')
    
    # Create detector and find anomalies
    detector = SimpleAnomalyDetector()
    anomalies = detector.detect_anomalies(df)
    
    # Print results
    print("\n=== ANOMALY DETECTION RESULTS ===")
    print(f"\nFound {len(anomalies)} anomalous sessions")
    
    for i, anomaly in enumerate(anomalies, 1):
        print(f"\nAnomaly #{i}:")
        print(f"User: {anomaly['user_id']}")
        print(f"Session ID: {anomaly['session_id']}")
        print(f"Start Time: {anomaly['start_time']}")
        print(f"Duration: {anomaly['duration']:.1f} minutes")
        print(f"Number of Actions: {anomaly['n_actions']}")
        print(f"Unique IPs: {anomaly['n_unique_ips']}")
        print(f"Failure Rate: {anomaly['failure_rate']:.1%}")
        print("\nReasons for flagging:")
        for reason in anomaly['reasons']:
            print(f"- {reason}")

if __name__ == "__main__":
    main() 