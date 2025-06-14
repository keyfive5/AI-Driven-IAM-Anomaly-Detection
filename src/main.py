import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Tuple, List
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from data_generator import IAMLogGenerator
from feature_engineering import FeatureEngineer
from models.hybrid_model import HybridAnomalyDetector
from data.iam_log_reader import get_log_reader, AWSCloudTrailReader # Import necessary reader classes

class AnomalyDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IAM Anomaly Detection")
        self.root.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel (left side)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Use grid for controls within the control_frame
        current_row = 0

        # Data Source Selection
        ttk.Label(self.control_frame, text="Data Source:").grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.data_source_var = tk.StringVar(value="Synthetic Data")
        self.data_source_options = ["Synthetic Data", "Real AWS CloudTrail Logs"]
        self.data_source_combobox = ttk.Combobox(self.control_frame, textvariable=self.data_source_var, values=self.data_source_options, state="readonly")
        self.data_source_combobox.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.data_source_combobox.bind("<<ComboboxSelected>>", self.on_data_source_change) # Bind event
        current_row += 1

        # Synthetic Data Controls (initially visible)
        self.synthetic_controls = []

        label = ttk.Label(self.control_frame, text="Number of Events:")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.synthetic_controls.append(label)
        self.n_events = ttk.Spinbox(self.control_frame, from_=100, to=5000, width=10)
        self.n_events.set(500)
        self.n_events.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.synthetic_controls.append(self.n_events)
        current_row += 1
        
        label = ttk.Label(self.control_frame, text="Number of Users:")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.synthetic_controls.append(label)
        self.n_users = ttk.Spinbox(self.control_frame, from_=10, to=200, width=10)
        self.n_users.set(20)
        self.n_users.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.synthetic_controls.append(self.n_users)
        current_row += 1
        
        label = ttk.Label(self.control_frame, text="Number of Roles:")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.synthetic_controls.append(label)
        self.n_roles = ttk.Spinbox(self.control_frame, from_=2, to=20, width=10)
        self.n_roles.set(3)
        self.n_roles.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.synthetic_controls.append(self.n_roles)
        current_row += 1
        
        label = ttk.Label(self.control_frame, text="Number of Actions:")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.synthetic_controls.append(label)
        self.n_actions = ttk.Spinbox(self.control_frame, from_=5, to=50, width=10)
        self.n_actions.set(5)
        self.n_actions.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.synthetic_controls.append(self.n_actions)
        current_row += 1
        
        ttk.Label(self.control_frame, text="Contamination Ratio:").grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.contamination_ratio = ttk.Spinbox(self.control_frame, from_=0.01, to=0.5, increment=0.01, width=10, format="%.2f")
        self.contamination_ratio.set(0.10) # Default from HybridAnomalyDetector
        self.contamination_ratio.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        # Add run button
        self.run_button = ttk.Button(self.control_frame, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=current_row, column=0, columnspan=2, pady=20)
        current_row += 1

        # --- Dedicated Status and Progress Area (using grid internally) ---
        self.status_progress_frame = ttk.Frame(self.control_frame) # New frame
        self.status_progress_frame.grid(row=current_row, column=0, columnspan=2, sticky="nsew", pady=10)
        
        # Configure grid weights for this frame to allow it to expand
        self.control_frame.grid_rowconfigure(current_row, weight=1) # Allow this row to expand vertically
        self.control_frame.grid_columnconfigure(0, weight=1) # Allow columns to expand horizontally
        self.control_frame.grid_columnconfigure(1, weight=1)

        # Internal grid for status_progress_frame
        self.status_progress_frame.grid_columnconfigure(0, weight=1) # Allow column 0 to expand

        inner_row = 0
        self.progress_label = ttk.Label(self.status_progress_frame, text="Progress: 0%")
        self.progress_label.grid(row=inner_row, column=0, sticky="w", pady=5)
        inner_row += 1
        
        self.progress_bar = ttk.Progressbar(self.status_progress_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(row=inner_row, column=0, sticky="ew", pady=5, padx=10)
        inner_row += 1

        self.status_text = tk.Text(self.status_progress_frame, height=8, width=30)
        self.status_text.grid(row=inner_row, column=0, sticky="nsew", pady=10)
        self.status_progress_frame.grid_rowconfigure(inner_row, weight=1) # Allow text area to expand vertically
        # --- End Status and Progress Area ---

        # Create visualization frame (right side)
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualization")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize data storage
        self.df = None
        self.predictions = None
        self.scores = None

        # Initial call to adjust controls
        self.on_data_source_change() # Call once to set initial state
        
    def on_data_source_change(self, event=None):
        selected_source = self.data_source_var.get()
        if selected_source == "Synthetic Data":
            for widget in self.synthetic_controls:
                widget.grid()
            self.n_events.config(state="enabled")
            self.n_users.config(state="enabled")
            self.n_roles.config(state="enabled")
            self.n_actions.config(state="enabled")
        else: # Real Logs selected
            for widget in self.synthetic_controls:
                widget.grid_remove() # Hide widgets
            self.n_events.config(state="disabled")
            self.n_users.config(state="disabled")
            self.n_roles.config(state="disabled")
            self.n_actions.config(state="disabled")
    
    def update_status(self, message):
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks() # Use update_idletasks for smoother updates
    
    def _update_progress_bar(self, value, message=""):
        self.progress_bar['value'] = value
        self.progress_label.config(text=f"Progress: {value}%")
        if message: # Optionally update status text with progress message
            self.update_status(message)
        self.root.update_idletasks() # Force GUI update

    def update_visualization(self):
        self.figure.clear()
        
        # Create 2x2 subplot layout
        ax1 = self.figure.add_subplot(221)
        ax2 = self.figure.add_subplot(222)
        ax3 = self.figure.add_subplot(223)
        ax4 = self.figure.add_subplot(224)
        
        # Plot 1: Anomaly Score Distribution
        ax1.hist(self.scores, bins=50)
        ax1.set_title('Distribution of Anomaly Scores')
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Count')
        
        # Plot 2: Anomalies by Hour
        if 'timestamp' in self.df.columns and not self.df['timestamp'].empty: # Check if timestamp exists and is not empty
            self.df['hour'] = self.df['timestamp'].dt.hour
            anomaly_hours = self.df[self.predictions == 1]['hour'].value_counts().sort_index()
            if not anomaly_hours.empty:
                ax2.bar(anomaly_hours.index, anomaly_hours.values)
                ax2.set_title('Anomalies by Hour of Day')
                ax2.set_xlabel('Hour')
                ax2.set_ylabel('Number of Anomalies')
            else:
                ax2.text(0.5, 0.5, 'No anomalies to plot', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
                ax2.set_title('Anomalies by Hour of Day')
        else:
            ax2.text(0.5, 0.5, 'Timestamp data not available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_title('Anomalies by Hour of Day')

        # Plot 3: Anomalies by User
        if 'user_id' in self.df.columns and not self.df['user_id'].empty:
            anomaly_users = self.df[self.predictions == 1]['user_id'].value_counts().head(10)
            if not anomaly_users.empty:
                ax3.bar(range(len(anomaly_users)), anomaly_users.values)
                ax3.set_title('Top 10 Users with Anomalies')
                ax3.set_xlabel('User ID')
                ax3.set_ylabel('Number of Anomalies')
                ax3.set_xticks(range(len(anomaly_users)))
                ax3.set_xticklabels(anomaly_users.index, rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No anomalies to plot', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
                ax3.set_title('Top 10 Users with Anomalies')
        else:
            ax3.text(0.5, 0.5, 'User ID data not available', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            ax3.set_title('Top 10 Users with Anomalies')

        # Plot 4: Anomaly Score vs Time
        if 'timestamp' in self.df.columns and not self.df['timestamp'].empty:
            ax4.scatter(self.df['timestamp'], self.scores, c=self.predictions, cmap='coolwarm', alpha=0.6)
            ax4.set_title('Anomaly Scores Over Time')
            ax4.set_xlabel('Timestamp')
            ax4.set_ylabel('Anomaly Score')
        else:
            ax4.text(0.5, 0.5, 'Timestamp data not available', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
            ax4.set_title('Anomaly Scores Over Time')

        self.figure.tight_layout()
        self.canvas.draw()
        
    def run_analysis(self):
        self.run_button.state(['disabled'])
        self.status_text.delete(1.0, tk.END)
        self.progress_bar['value'] = 0 # Reset progress bar
        self.progress_label.config(text="Progress: 0%") # Reset label
        self.root.update() # Force an immediate update to show reset progress bar and label
        
        # This function will run in a separate thread
        def analysis_thread():
            try:
                selected_source = self.data_source_var.get()
                df_local = None
                true_anomalies_exist = False

                if selected_source == "Synthetic Data":
                    self.root.after(0, self.update_status, "Generating IAM logs... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing data generator...")
                    generator = IAMLogGenerator(
                        n_users=int(self.n_users.get()),
                        n_roles=int(self.n_roles.get()),
                        n_actions=int(self.n_actions.get())
                    )
                    self.root.after(0, self._update_progress_bar, 10, "Generating synthetic dataset...")
                    df_local = generator.generate_dataset(n_events=int(self.n_events.get()), anomaly_ratio=0.1)
                    self.root.after(0, self._update_progress_bar, 20, "Data generation complete!")
                    true_anomalies_exist = True # Synthetic data has known anomalies
                elif selected_source == "Real AWS CloudTrail Logs":
                    self.root.after(0, self.update_status, "Loading real AWS CloudTrail logs... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing log reader...")
                    reader = get_log_reader('aws')
                    log_file_path = "data/sample_aws_cloudtrail.json"
                    if not os.path.exists(log_file_path):
                        self.root.after(0, self.update_status, f"Error: Log file not found at {log_file_path}")
                        self.root.after(0, lambda: self.run_button.state(['!disabled']))
                        return
                    
                    self.root.after(0, self._update_progress_bar, 10, f"Reading logs from {log_file_path}...")
                    df_local = reader.read_logs(log_file_path)
                    
                    # Validate and clean logs
                    if not reader.validate_logs(df_local):
                        self.root.after(0, self.update_status, "Error: Real logs failed validation.")
                        self.root.after(0, lambda: self.run_button.state(['!disabled']))
                        return
                    df_local = reader.clean_logs(df_local)
                    self.root.after(0, self._update_progress_bar, 20, "Real logs loaded and cleaned!")

                    # For real logs, we don't have ground truth, so set is_anomaly to 0 for evaluation purposes
                    # The model will still detect anomalies based on its training, but performance metrics won't be truly indicative.
                    df_local['is_anomaly'] = 0
                    true_anomalies_exist = False # No known true anomalies for evaluation

                if df_local is None or df_local.empty:
                    self.root.after(0, self.update_status, "Error: No data to process.")
                    self.root.after(0, lambda: self.run_button.state(['!disabled']))
                    return
                
                # Extract features (40% progress)
                self.root.after(0, self.update_status, "Extracting features... (2/4)")
                self.root.after(0, self._update_progress_bar, 25, "Initializing feature engineer...")
                engineer = FeatureEngineer()
                self.root.after(0, self._update_progress_bar, 30, "Applying feature engineering...")
                df_features_local = engineer.engineer_features(df_local)
                self.root.after(0, self._update_progress_bar, 40, "Feature extraction complete!")
                
                # Train model (70% progress, as this is typically the longest part)
                self.root.after(0, self.update_status, "Training hybrid anomaly detection model... (3/4)")
                # The actual progress will be managed by the detector's internal callbacks
                detector = HybridAnomalyDetector(contamination=float(self.contamination_ratio.get()))
                # Pass the _update_progress_bar as a callback to the detector's fit method
                detector.fit(df_features_local, engineer.get_feature_columns(), progress_callback=self._update_progress_bar)
                # No fixed progress update here, as the detector manages it internally from 45% to 70%
                
                # Making predictions (90% progress)
                self.root.after(0, self.update_status, "Making predictions... (4/4)")
                self.root.after(0, self._update_progress_bar, 75, "Generating predictions...")
                predictions_local, scores_local = detector.predict(df_features_local)
                self.root.after(0, self._update_progress_bar, 90, "Prediction complete!")
                
                # Calculate metrics and save results (100% progress)
                self.root.after(0, self.update_status, "Calculating performance metrics...")
                
                if true_anomalies_exist:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(df_local['is_anomaly'], predictions_local)
                    recall = recall_score(df_local['is_anomaly'], predictions_local)
                    f1 = f1_score(df_local['is_anomaly'], predictions_local)
                    
                    self.root.after(0, self.update_status, "\nModel Performance:")
                    self.root.after(0, self.update_status, f"Contamination Ratio: {self.contamination_ratio.get()}")
                    self.root.after(0, self.update_status, f"Precision: {precision:.3f}")
                    self.root.after(0, self.update_status, f"Recall: {recall:.3f}")
                    self.root.after(0, self.update_status, f"F1 Score: {f1:.3f}")
                else:
                    self.root.after(0, self.update_status, "\nModel Performance: (Note: Metrics not applicable for unlabeled real logs)")
                    self.root.after(0, self.update_status, f"Contamination Ratio: {self.contamination_ratio.get()}")
                    self.root.after(0, self.update_status, f"Detected Anomalies: {predictions_local.sum()}")
                
                self.root.after(0, self.update_status, "Saving results...")
                os.makedirs('output', exist_ok=True)
                results_df = df_local.copy()
                results_df['predicted_anomaly'] = predictions_local
                results_df['anomaly_score'] = scores_local
                results_df.to_csv('output/anomaly_results.csv', index=False)
                
                self.root.after(0, self.update_status, "\nResults saved to 'output/anomaly_results.csv'")
                
                self.root.after(0, self._update_progress_bar, 100, "Analysis Complete!") # Final update to 100% with message

                # Pass results to main thread for visualization
                self.root.after(0, self.finalize_analysis, df_local, predictions_local, scores_local)
                
            except Exception as e:
                self.root.after(0, self.update_status, f"Error: {str(e)}")
            finally:
                self.root.after(0, lambda: self.run_button.state(['!disabled']))
        
        # Start analysis in a separate thread
        threading.Thread(target=analysis_thread, daemon=True).start()

    def finalize_analysis(self, df, predictions, scores):
        self.df = df
        self.predictions = predictions
        self.scores = scores
        self.update_visualization()

def main():
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 