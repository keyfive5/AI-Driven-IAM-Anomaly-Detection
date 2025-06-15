import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple
import json

class IAMLogGenerator:
    def __init__(self, n_users: int = 100, n_roles: int = 5, n_actions: int = 10):
        self.n_users = n_users
        self.n_roles = n_roles
        self.n_actions = n_actions
        
        # Initialize user roles and permissions
        self.roles = [f"role_{i}" for i in range(n_roles)]
        self.actions = [f"action_{i}" for i in range(n_actions)]
        self.users = [f"user_{i}" for i in range(n_users)]
        
        # Assign roles to users
        self.user_roles = {user: random.choice(self.roles) for user in self.users}
        
        # Define role-action permissions (ensure k is not larger than available actions)
        self.role_permissions = {
            role: set(random.sample(self.actions, k=min(random.randint(3, 8), len(self.actions))))
            for role in self.roles
        }
        
        # Define normal working hours (9 AM to 5 PM)
        self.working_hours = (9, 17)
        
        # Define common user agents
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        # Define common resources
        self.resources = [
            "database",
            "file_server",
            "api_gateway",
            "storage_bucket",
            "application_server"
        ]
        
        # Define regions
        self.regions = [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "ap-southeast-1"
        ]
        
        # Define user behavior patterns
        self.user_patterns = {
            'normal': {
                'login_frequency': (1, 3),  # logins per day
                'action_frequency': (5, 20),  # actions per login
                'session_duration': (30, 180),  # minutes
                'ip_change_probability': 0.1
            },
            'anomalous': {
                'login_frequency': (5, 10),
                'action_frequency': (30, 100),
                'session_duration': (300, 600),
                'ip_change_probability': 0.8
            }
        }
        
    def generate_user_session(self, user: str, is_anomaly: bool = False) -> List[Dict]:
        """Generate a sequence of actions for a user session."""
        pattern = self.user_patterns['anomalous'] if is_anomaly else self.user_patterns['normal']
        
        # Generate session start time
        session_start = datetime.now() - timedelta(days=random.randint(0, 30))
        if not is_anomaly:
            # Normal sessions during working hours
            session_start = session_start.replace(
                hour=random.randint(self.working_hours[0], self.working_hours[1]),
                minute=random.randint(0, 59)
            )
        else:
            # Anomalous sessions at any time
            session_start = session_start.replace(
                hour=random.randint(0, 23),
                minute=random.randint(0, 59)
            )
        
        # Generate session duration
        session_duration = random.randint(*pattern['session_duration'])
        session_end = session_start + timedelta(minutes=session_duration)
        
        # Generate number of actions
        n_actions = random.randint(*pattern['action_frequency'])
        
        # Generate actions
        actions = []
        role = self.user_roles[user]
        current_time = session_start
        
        # Generate initial IP
        ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        for _ in range(n_actions):
            # Determine if IP should change (more likely for anomalies)
            if random.random() < pattern['ip_change_probability']:
                if is_anomaly:
                    ip = f"203.0.{random.randint(1, 254)}.{random.randint(1, 254)}"
                else:
                    ip = f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
            
            # Generate action
            if is_anomaly and random.random() < 0.3:
                # Anomalous action: not permitted for role
                invalid_actions = set(self.actions) - self.role_permissions[role]
                if invalid_actions:
                    action = random.choice(list(invalid_actions))
                else:
                    action = random.choice(self.actions)
            else:
                # Normal action: permitted for role
                action = random.choice(list(self.role_permissions[role]))
            
            # Generate timestamp within session
            time_delta = random.randint(0, session_duration)
            timestamp = session_start + timedelta(minutes=time_delta)
            
            # Generate resource
            resource = random.choice(self.resources)
            
            # Generate user agent
            user_agent = random.choice(self.user_agents)
            
            # Generate region
            region = random.choice(self.regions)
            
            actions.append({
                'timestamp': timestamp,
                'user_id': user,
                'role': role,
                'action': action,
                'ip_address': ip,
                'status': 'success' if not is_anomaly or random.random() < 0.7 else 'failure',
                'resource': resource,
                'user_agent': user_agent,
                'region': region,
                'session_id': f"{user}_{session_start.strftime('%Y%m%d%H%M%S')}",
                'session_start': session_start,
                'session_end': session_end,
                'is_anomaly': is_anomaly
            })
            
            current_time = timestamp
        
        return actions
    
    def generate_normal_log(self, n_events: int) -> pd.DataFrame:
        """Generate normal IAM logs following typical patterns."""
        logs = []
        # Generate sessions until n_events is met
        current_events = 0
        while current_events < n_events:
            user = random.choice(self.users)
            session_logs = self.generate_user_session(user, is_anomaly=False)
            # Ensure we don't exceed n_events significantly
            if current_events + len(session_logs) > n_events:
                session_logs = session_logs[:n_events - current_events]
            logs.extend(session_logs)
            current_events += len(session_logs)
        
        df = pd.DataFrame(logs)
        # Add a unique _temp_row_id at generation time
        df['_temp_row_id'] = range(len(df)) # Ensure unique, contiguous IDs
        return df
    
    def inject_anomalies(self, df: pd.DataFrame, anomaly_ratio: float = 0.1) -> pd.DataFrame:
        """Inject various types of anomalies into the logs."""
        n_anomalies = int(len(df) * anomaly_ratio)
        # Ensure we have enough unique indices to sample from without replacement
        available_indices = df.index.tolist()
        if n_anomalies > len(available_indices):
            n_anomalies = len(available_indices) # Adjust if anomaly_ratio is too high for current df size
            
        anomaly_indices = random.sample(available_indices, n_anomalies)
        
        for idx in anomaly_indices:
            anomaly_type = random.choice(['time', 'role', 'ip', 'action', 'session'])
            
            if anomaly_type == 'time':
                # Anomaly: Access outside working hours
                # Ensure timestamp is datetime before manipulation
                original_timestamp = df.at[idx, 'timestamp']
                if pd.isna(original_timestamp):
                    continue # Skip if timestamp is invalid
                hour = random.choice([random.randint(0, 8), random.randint(18, 23)])
                df.at[idx, 'timestamp'] = original_timestamp.replace(hour=hour)
                
            elif anomaly_type == 'role':
                # Anomaly: Action not permitted for role
                role = df.at[idx, 'role']
                invalid_actions = set(self.actions) - self.role_permissions[role]
                if invalid_actions:
                    df.at[idx, 'action'] = random.choice(list(invalid_actions))
                    
            elif anomaly_type == 'ip':
                # Anomaly: External IP address
                df.at[idx, 'ip_address'] = f"203.0.{random.randint(1, 254)}.{random.randint(1, 254)}"
                
            elif anomaly_type == 'action':
                # Anomaly: Unusual action frequency (by changing the action to a less common one or highly privileged one)
                # For now, just change to a random action not typically associated with the role for simplicity
                role = df.at[idx, 'role']
                all_actions_except_current = [act for act in self.actions if act != df.at[idx, 'action']]
                if all_actions_except_current:
                    df.at[idx, 'action'] = random.choice(all_actions_except_current)
                
            elif anomaly_type == 'session':
                # Anomaly: Unusually long session
                session_start = df.at[idx, 'session_start']
                if pd.isna(session_start):
                    continue # Skip if session_start is invalid
                session_end = session_start + timedelta(hours=random.randint(4, 12))
                df.at[idx, 'session_end'] = session_end
                
            df.at[idx, 'is_anomaly'] = True # Set to boolean True
            
        return df
    
    def generate_dataset(self, n_events: int = 10000, anomaly_ratio: float = 0.1) -> pd.DataFrame:
        """Generate a complete dataset with normal and anomalous events."""
        if n_events <= 0: return pd.DataFrame() # Handle empty case

        # Generate normal logs based on a portion of total events
        n_normal = int(n_events * (1 - anomaly_ratio))
        if n_normal <= 0: n_normal = 1 # Ensure at least one normal event if n_events > 0
        
        normal_logs = self.generate_normal_log(n_normal)
        
        # Generate anomalous logs for the remaining events. These will be added directly and marked as anomalies.
        n_anomalous = n_events - len(normal_logs) # Remaining events to generate as anomalies
        
        anomalous_logs_list = []
        for _ in range(n_anomalous):
            user = random.choice(self.users)
            session_logs = self.generate_user_session(user, is_anomaly=True)
            anomalous_logs_list.extend(session_logs)
        
        if anomalous_logs_list:
            anomalous_df = pd.DataFrame(anomalous_logs_list)
            # Add _temp_row_id to anomalous_df as well, ensuring unique IDs across the combined dataset
            # Start from the max ID of normal_logs to prevent overlap
            start_id = normal_logs['_temp_row_id'].max() + 1 if '_temp_row_id' in normal_logs.columns and not normal_logs.empty else 0
            anomalous_df['_temp_row_id'] = range(start_id, start_id + len(anomalous_df))
            # Set is_anomaly to True for all injected anomalous events
            anomalous_df['is_anomaly'] = True
            combined_df = pd.concat([normal_logs, anomalous_df], ignore_index=True)
        else:
            combined_df = normal_logs.copy()

        # Ensure the final dataframe has exactly n_events by sampling if needed
        if len(combined_df) > n_events:
            combined_df = combined_df.sample(n=n_events, random_state=42).reset_index(drop=True)
        elif len(combined_df) < n_events: # If we generated less than n_events, generate more normal logs
            remaining_events = n_events - len(combined_df)
            additional_normal_logs = self.generate_normal_log(remaining_events) # This will also add _temp_row_id
            # Adjust _temp_row_id for new logs to be unique
            start_id = combined_df['_temp_row_id'].max() + 1 if '_temp_row_id' in combined_df.columns and not combined_df.empty else 0
            additional_normal_logs['_temp_row_id'] = range(start_id, start_id + len(additional_normal_logs))
            combined_df = pd.concat([combined_df, additional_normal_logs], ignore_index=True)
        
        # Final check for is_anomaly dtype, ensure it's boolean
        combined_df['is_anomaly'] = combined_df['is_anomaly'].astype(bool)

        # Ensure timestamps are strictly increasing within sessions after all manipulations
        # This requires re-sorting and potentially adjusting timestamps to enforce strict increase
        # For simplicity in generation, we sort here. More complex scenarios might need iterative adjustment.
        combined_df = combined_df.sort_values(by=['user_id', 'session_id', 'timestamp']).reset_index(drop=True)

        return combined_df
    
    def save_dataset(self, df: pd.DataFrame, file_path: str):
        """Save the generated dataset to a file."""
        df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    generator = IAMLogGenerator()
    df = generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
    print(f"Generated dataset with {len(df)} events")
    print(f"Number of anomalies: {df['is_anomaly'].sum()}")
    print("\nSample of generated data:")
    print(df.head())
    
    # Save the dataset
    generator.save_dataset(df, "output/iam_logs.csv") 