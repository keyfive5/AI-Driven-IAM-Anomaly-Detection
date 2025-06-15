import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, time
import numpy as np
from simple_detector import SimpleAnomalyDetector

def create_visualizations(df, anomalies):
    # Convert timestamps if needed
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['session_start'] = pd.to_datetime(df['session_start'])
        df['session_end'] = pd.to_datetime(df['session_end'])

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Session Duration vs Actions',
            'IP Changes Over Time',
            'Failure Rate Distribution',
            'Actions by Hour of Day',
            'Session Duration Distribution',
            'IP Changes Distribution'
        )
    )

    # 1. Session Duration vs Actions scatter plot
    normal_sessions = df[~df['session_id'].isin([a['session_id'] for a in anomalies])]
    anomaly_sessions = df[df['session_id'].isin([a['session_id'] for a in anomalies])]
    
    # Calculate metrics for normal sessions
    normal_metrics = normal_sessions.groupby('session_id').agg({
        'session_start': 'first',
        'session_end': 'first',
        'ip_address': 'nunique',
        'status': lambda x: (x == 'failure').mean()
    }).reset_index()
    
    normal_metrics['duration'] = (normal_metrics['session_end'] - normal_metrics['session_start']).dt.total_seconds() / 60
    normal_metrics['n_actions'] = normal_sessions.groupby('session_id').size().values

    # Add normal sessions
    fig.add_trace(
        go.Scatter(
            x=normal_metrics['duration'],
            y=normal_metrics['n_actions'],
            mode='markers',
            name='Normal Sessions',
            marker=dict(color='blue', size=8)
        ),
        row=1, col=1
    )

    # Add anomalous sessions
    for anomaly in anomalies:
        fig.add_trace(
            go.Scatter(
                x=[anomaly['duration']],
                y=[anomaly['n_actions']],
                mode='markers',
                name='Anomalous Session',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"User: {anomaly['user_id']}<br>Reasons: {', '.join(anomaly['reasons'])}"],
                hoverinfo='text'
            ),
            row=1, col=1
        )

    # 2. IP Changes Over Time
    for anomaly in anomalies:
        anomaly_data = df[df['session_id'] == anomaly['session_id']]
        cumulative_ips = []
        seen_ips = set()
        for ip in anomaly_data['ip_address']:
            seen_ips.add(ip)
            cumulative_ips.append(len(seen_ips))
            
        fig.add_trace(
            go.Scatter(
                x=anomaly_data['timestamp'],
                y=cumulative_ips,
                mode='lines+markers',
                name=f'Anomaly IP Changes',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ),
            row=1, col=2
        )

    # 3. Failure Rate Distribution
    fig.add_trace(
        go.Histogram(
            x=normal_metrics['status'],
            name='Normal Sessions',
            marker_color='blue',
            opacity=0.7
        ),
        row=2, col=1
    )

    for anomaly in anomalies:
        fig.add_trace(
            go.Scatter(
                x=[anomaly['failure_rate']],
                y=[0],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"User: {anomaly['user_id']}<br>Failure Rate: {anomaly['failure_rate']:.1%}"],
                hoverinfo='text'
            ),
            row=2, col=1
        )

    # 4. Actions by Hour of Day
    df['hour'] = df['timestamp'].dt.hour
    hourly_actions = df.groupby('hour').size()
    
    fig.add_trace(
        go.Bar(
            x=hourly_actions.index,
            y=hourly_actions.values,
            name='Normal Actions',
            marker_color='blue',
            opacity=0.7
        ),
        row=2, col=2
    )

    # Add anomalous actions
    for anomaly in anomalies:
        anomaly_data = df[df['session_id'] == anomaly['session_id']]
        anomaly_hours = anomaly_data.groupby(anomaly_data['timestamp'].dt.hour).size()
        
        fig.add_trace(
            go.Scatter(
                x=anomaly_hours.index,
                y=anomaly_hours.values,
                mode='markers',
                name='Anomaly Actions',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"User: {anomaly['user_id']}<br>Hour: {hour}" for hour in anomaly_hours.index],
                hoverinfo='text'
            ),
            row=2, col=2
        )

    # 5. Session Duration Distribution
    fig.add_trace(
        go.Histogram(
            x=normal_metrics['duration'],
            name='Normal Sessions',
            marker_color='blue',
            opacity=0.7
        ),
        row=3, col=1
    )

    for anomaly in anomalies:
        fig.add_trace(
            go.Scatter(
                x=[anomaly['duration']],
                y=[0],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"User: {anomaly['user_id']}<br>Duration: {anomaly['duration']:.1f} minutes"],
                hoverinfo='text'
            ),
            row=3, col=1
        )

    # 6. IP Changes Distribution
    fig.add_trace(
        go.Histogram(
            x=normal_metrics['ip_address'],
            name='Normal Sessions',
            marker_color='blue',
            opacity=0.7
        ),
        row=3, col=2
    )

    for anomaly in anomalies:
        fig.add_trace(
            go.Scatter(
                x=[anomaly['n_unique_ips']],
                y=[0],
                mode='markers',
                name='Anomaly',
                marker=dict(color='red', size=12, symbol='star'),
                text=[f"User: {anomaly['user_id']}<br>Unique IPs: {anomaly['n_unique_ips']}"],
                hoverinfo='text'
            ),
            row=3, col=2
        )

    # Update layout
    fig.update_layout(
        height=1200,
        width=1200,
        title_text="IAM Anomaly Detection Visualization",
        showlegend=True,
        template="plotly_white"
    )

    # Update axes labels
    fig.update_xaxes(title_text="Duration (minutes)", row=1, col=1)
    fig.update_yaxes(title_text="Number of Actions", row=1, col=1)
    fig.update_xaxes(title_text="Time", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative IP Changes", row=1, col=2)
    fig.update_xaxes(title_text="Failure Rate", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Hour of Day", row=2, col=2)
    fig.update_yaxes(title_text="Number of Actions", row=2, col=2)
    fig.update_xaxes(title_text="Duration (minutes)", row=3, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    fig.update_xaxes(title_text="Number of Unique IPs", row=3, col=2)
    fig.update_yaxes(title_text="Count", row=3, col=2)

    # Save the figure
    fig.write_html("anomaly_visualization.html")
    print("Visualization saved to anomaly_visualization.html")

def main():
    # Read the test data
    df = pd.read_csv('test_data.csv')
    
    # Create detector and find anomalies
    detector = SimpleAnomalyDetector()
    anomalies = detector.detect_anomalies(df)
    
    # Create visualizations
    create_visualizations(df, anomalies)

if __name__ == "__main__":
    main() 