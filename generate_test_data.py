from src.data_generator import IAMLogGenerator
import pandas as pd

def generate_test_data():
    # Create generator with fewer users and actions for simplicity
    generator = IAMLogGenerator(n_users=5, n_roles=3, n_actions=5)
    
    # Generate 99 normal logs
    normal_logs = generator.generate_normal_log(n_events=99)
    
    # Generate 1 anomalous log with a clear anomaly (outside working hours)
    anomalous_user = generator.users[0]  # Use the first user
    anomalous_session = generator.generate_user_session(anomalous_user, is_anomaly=True)
    anomalous_df = pd.DataFrame(anomalous_session)
    
    # Combine normal and anomalous logs
    combined_logs = pd.concat([normal_logs, anomalous_df], ignore_index=True)
    
    # Sort by timestamp
    combined_logs = combined_logs.sort_values('timestamp')
    
    # Save to CSV
    combined_logs.to_csv('test_data.csv', index=False)
    print(f"Generated {len(combined_logs)} logs with 1 anomaly")
    print("Data saved to test_data.csv")

if __name__ == "__main__":
    generate_test_data() 