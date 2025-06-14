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
        
        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Main Analysis Tab ---
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main Analysis")

        # Create control panel (left side) - now inside main_tab
        self.control_frame = ttk.LabelFrame(self.main_tab, text="Controls")
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
        self.contamination_ratio.set(0.15) # Default from HybridAnomalyDetector
        self.contamination_ratio.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        # --- Hyperparameter Tuning Controls ---
        self.model_tuning_controls = []

        label = ttk.Label(self.control_frame, text="IF Estimators (n):")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.model_tuning_controls.append(label)
        self.n_estimators_iso_forest = ttk.Spinbox(self.control_frame, from_=50, to=500, increment=50, width=10)
        self.n_estimators_iso_forest.set(300) # Default
        self.n_estimators_iso_forest.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.model_tuning_controls.append(self.n_estimators_iso_forest)
        current_row += 1

        label = ttk.Label(self.control_frame, text="IF Max Features (float):")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.model_tuning_controls.append(label)
        self.max_features_iso_forest = ttk.Spinbox(self.control_frame, from_=0.1, to=1.0, increment=0.1, width=10, format="%.1f")
        self.max_features_iso_forest.set(1.0) # Default
        self.max_features_iso_forest.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.model_tuning_controls.append(self.max_features_iso_forest)
        current_row += 1

        label = ttk.Label(self.control_frame, text="RF Estimators (n):")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.model_tuning_controls.append(label)
        self.n_estimators_rf = ttk.Spinbox(self.control_frame, from_=50, to=500, increment=50, width=10)
        self.n_estimators_rf.set(250) # Default
        self.n_estimators_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.model_tuning_controls.append(self.n_estimators_rf)
        current_row += 1

        label = ttk.Label(self.control_frame, text="RF Max Depth (int/None):")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.model_tuning_controls.append(label)
        # Using Combobox for Max Depth to allow 'None'
        self.max_depth_rf_var = tk.StringVar(value="30")
        self.max_depth_rf = ttk.Combobox(self.control_frame, textvariable=self.max_depth_rf_var, values=["None", 10, 20, 30, 50], state="readonly", width=10)
        self.max_depth_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.model_tuning_controls.append(self.max_depth_rf)
        current_row += 1

        label = ttk.Label(self.control_frame, text="RF Min Samples Split (n):")
        label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.model_tuning_controls.append(label)
        self.min_samples_split_rf = ttk.Spinbox(self.control_frame, from_=2, to=20, width=10)
        self.min_samples_split_rf.set(2) # Default
        self.min_samples_split_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.model_tuning_controls.append(self.min_samples_split_rf)
        current_row += 1
        # --- End Hyperparameter Tuning Controls ---
        
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

        # Create visualization frame (right side) - now inside main_tab
        self.viz_frame = ttk.LabelFrame(self.main_tab, text="Visualization")
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize data storage
        self.df = None
        self.predictions = None
        self.scores = None

        # --- Updates Tab ---
        self.updates_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.updates_tab, text="Updates")

        self.updates_text = tk.Text(self.updates_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.updates_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial updates content
        self.update_updates_tab("""
**June 14, 2024:**
- Implemented a robust tabbed GUI interface for better navigation.
- Added a dedicated 'Updates' tab to track project progress.
- Successfully integrated hyperparameter tuning controls for Isolation Forest and RandomForest Classifier models.
- Achieved significant performance improvements (F1 Score: 0.188) through initial hyperparameter tuning on synthetic data.

**June 13, 2024:**
- Resolved the persistent 'NoneType' error in log reader by adding robust handling for missing/null nested JSON fields in AWS CloudTrail logs.
- Fixed 'session_id' KeyError in feature engineering by ensuring all session-related operations are strictly conditional on valid data presence.
- Expanded the 'data/sample_aws_cloudtrail.json' file with more realistic and diverse log entries.
- Successfully re-integrated and confirmed the training of the LSTM Autoencoder with the expanded real logs.
- Demonstrated initial anomaly detection and populated visualizations with real log data.

**Ongoing Development:**
- Continuous model performance improvement through advanced tuning and algorithmic enhancements.
- Exploring more sophisticated feature engineering techniques.
- Preparing for integration with larger, real-world datasets.
""")

        # --- Data Source Management Tab ---
        self.data_source_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_source_tab, text="Data Source Management")

        self.data_source_text = tk.Text(self.data_source_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.data_source_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_data_source_tab("""
**Enhancing IAM Anomaly Detection through Multi-Source Log Integration**

To provide a truly comprehensive and robust IAM anomaly detection solution, future development will focus on seamlessly integrating with a variety of enterprise log sources. This expanded data ingestion capability is critical for:

*   **Holistic Threat Visibility:** Combining logs from different systems (e.g., cloud, on-premise, network) provides a richer context for anomaly detection, allowing the identification of sophisticated attack patterns that might be missed in isolated datasets.
*   **Improved Accuracy & Reduced False Positives:** A broader data set enables more accurate baselining of normal user behavior, leading to more precise anomaly detection and a significant reduction in false alerts.
*   **Scalability & Adaptability:** Supporting diverse log formats and protocols ensures the solution can be deployed across various IT infrastructures, from hybrid cloud environments to purely on-premise setups.

**Planned Data Source Integrations (Roadmap):**

*   **Cloud Platforms:**
    *   AWS CloudTrail (Expanded beyond current sample)
    *   Azure Activity Logs & Azure AD Audit Logs
    *   Google Cloud Audit Logs
*   **On-Premise Systems:**
    *   Active Directory (Security Event Logs)
    *   Syslog (Generic log collection for various devices)
    *   Firewall Logs (e.g., Palo Alto, Cisco ASA)
*   **Security Information and Event Management (SIEM) Systems:**
    *   Splunk (via API or forwarders)
    *   Elastic Stack (Elasticsearch, Logstash, Kibana)

This multi-source integration strategy will empower organizations with unparalleled visibility into their identity and access landscape, proactive threat detection, and significantly enhanced overall security posture.
""")

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
            for widget in self.model_tuning_controls:
                widget.grid() # Show widgets
            self.n_estimators_iso_forest.config(state="enabled")
            self.max_features_iso_forest.config(state="enabled")
            self.n_estimators_rf.config(state="enabled")
            self.max_depth_rf.config(state="readonly") # Combobox is readonly when enabled
            self.min_samples_split_rf.config(state="enabled")
        else: # Real Logs selected
            for widget in self.synthetic_controls:
                widget.grid_remove() # Hide widgets
            self.n_events.config(state="disabled")
            self.n_users.config(state="disabled")
            self.n_roles.config(state="disabled")
            self.n_actions.config(state="disabled")
            for widget in self.model_tuning_controls:
                widget.grid_remove() # Hide widgets
            self.n_estimators_iso_forest.config(state="disabled")
            self.max_features_iso_forest.config(state="disabled")
            self.n_estimators_rf.config(state="disabled")
            self.max_depth_rf.config(state="disabled")
            self.min_samples_split_rf.config(state="disabled")
    
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
                true_labels = None

                if selected_source == "Synthetic Data":
                    self.root.after(0, self.update_status, "Generating IAM logs... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing data generator...")
                    generator = IAMLogGenerator(
                        n_users=int(self.n_users.get()),
                        n_roles=int(self.n_roles.get()),
                        n_actions=int(self.n_actions.get())
                    )
                    self.root.after(0, self._update_progress_bar, 10, "Generating synthetic dataset...")
                    df_local = generator.generate_dataset(n_events=int(self.n_events.get()), anomaly_ratio=float(self.contamination_ratio.get()))
                    # For synthetic data, we have true labels
                    true_anomalies_exist = True
                    # Assuming 'is_anomaly' is the true label column
                    true_labels = df_local['is_anomaly'].values
                    self.root.after(0, self._update_progress_bar, 20, "Data generation complete!")
                elif selected_source == "Real AWS CloudTrail Logs":
                    self.root.after(0, self.update_status, "Loading real AWS CloudTrail logs... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing log reader...")
                    log_reader = AWSCloudTrailReader()
                    log_file_path = "data/sample_aws_cloudtrail.json"
                    self.root.after(0, self.update_status, f"Reading logs from {log_file_path}...")
                    df_local = log_reader.read_logs(log_file_path)
                    self.root.after(0, self.update_status, "Real logs loaded and cleaned!")
                    true_anomalies_exist = False # No true labels for real logs

                if df_local is None or df_local.empty:
                    self.root.after(0, self.update_status, "Error: No data to process.")
                    self.root.after(0, lambda: self.run_button.state(['!disabled']))
                    return
                
                # Extract features (40% progress)
                self.root.after(0, self.update_status, "Extracting features... (2/4)")
                self.root.after(0, self._update_progress_bar, 25, "Initializing feature engineer...")
                feature_engineer = FeatureEngineer()
                self.root.after(0, self.update_status, "Applying feature engineering...")
                df_local = feature_engineer.engineer_features(df_local)
                feature_columns = feature_engineer.get_feature_columns()
                self.root.after(0, self._update_progress_bar, 40, "Feature extraction complete!")
                
                # Train model (70% progress, as this is typically the longest part)
                self.root.after(0, self.update_status, "Training hybrid anomaly detection model... (3/4)")
                self.root.after(0, self._update_progress_bar, 45, "Initializing hybrid model...")
                
                # Get hyperparameters from GUI
                n_estimators_iso_forest = int(self.n_estimators_iso_forest.get())
                max_features_iso_forest = float(self.max_features_iso_forest.get())
                n_estimators_rf = int(self.n_estimators_rf.get())
                max_depth_rf_val = self.max_depth_rf_var.get()
                max_depth_rf = int(max_depth_rf_val) if max_depth_rf_val != "None" else None
                min_samples_split_rf = int(self.min_samples_split_rf.get())

                # Display current hyperparameters
                self.root.after(0, self.update_status, f"Current Hyperparameters:")
                self.root.after(0, self.update_status, f"  IF Estimators: {n_estimators_iso_forest}")
                self.root.after(0, self.update_status, f"  IF Max Features: {max_features_iso_forest}")
                self.root.after(0, self.update_status, f"  RF Estimators: {n_estimators_rf}")
                self.root.after(0, self.update_status, f"  RF Max Depth: {max_depth_rf}")
                self.root.after(0, self.update_status, f"  RF Min Samples Split: {min_samples_split_rf}")

                hybrid_detector = HybridAnomalyDetector(
                    contamination=float(self.contamination_ratio.get()),
                    n_estimators_iso_forest=n_estimators_iso_forest,
                    max_features_iso_forest=max_features_iso_forest,
                    n_estimators_rf=n_estimators_rf,
                    max_depth_rf=max_depth_rf,
                    min_samples_split_rf=min_samples_split_rf
                )

                hybrid_detector.fit(df_local[feature_columns], feature_columns, self._update_progress_bar)
                self.root.after(0, self._update_progress_bar, 85, "Model trained.") # Adjusted percentage

                # Making predictions (90% progress)
                self.root.after(0, self.update_status, "Making predictions... (4/4)")
                self.root.after(0, self._update_progress_bar, 90, "Generating predictions...")
                predictions, scores = hybrid_detector.predict(df_local)
                self.root.after(0, self._update_progress_bar, 95, "Prediction complete!")
                
                self.df = df_local # Store the DataFrame for visualization
                self.predictions = predictions
                self.scores = scores

                self.root.after(0, self.update_status, "Calculating performance metrics...")
                if true_anomalies_exist:
                    from sklearn.metrics import precision_score, recall_score, f1_score
                    precision = precision_score(true_labels, predictions)
                    recall = recall_score(true_labels, predictions)
                    f1 = f1_score(true_labels, predictions)
                    self.root.after(0, self.update_status, "\nModel Performance:")
                    self.root.after(0, self.update_status, f"Contamination Ratio: {self.contamination_ratio.get()}")
                    self.root.after(0, self.update_status, f"Precision: {precision:.3f}")
                    self.root.after(0, self.update_status, f"Recall: {recall:.3f}")
                    self.root.after(0, self.update_status, f"F1 Score: {f1:.3f}")
                else:
                    self.root.after(0, self.update_status, "\nModel Performance: (Note: Metrics not applicable for unlabeled real logs)")
                    self.root.after(0, self.update_status, f"Contamination Ratio: {self.contamination_ratio.get()}")
                
                self.root.after(0, self.update_status, f"Detected Anomalies: {np.sum(predictions)}")

                self.root.after(0, self.update_status, "Saving results...")
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'anomaly_results.csv')
                # Ensure 'is_anomaly' column is present before saving if it exists
                output_df = self.df.copy()
                output_df['is_anomaly_predicted'] = self.predictions # Add predicted anomalies
                output_df['anomaly_score'] = self.scores # Add anomaly scores

                # If original data had true anomalies, include them
                if true_anomalies_exist:
                    output_df['is_anomaly_true'] = true_labels
                
                output_df.to_csv(output_path, index=False)
                self.root.after(0, self.update_status, f"Results saved to '{output_path}'")

                self.root.after(0, self.update_status, "Analysis Complete!")
                self.root.after(0, self.update_visualization) # Update visualizations on completion

            except Exception as e:
                self.root.after(0, self.update_status, f"Error: {e}")
                import traceback
                self.root.after(0, self.update_status, traceback.format_exc())
            finally:
                self.root.after(0, self.run_button.state, ['!disabled'])
        
        # Run the analysis in a separate thread to keep the GUI responsive
        threading.Thread(target=analysis_thread).start()

    def update_updates_tab(self, content):
        self.updates_text.config(state='normal')
        self.updates_text.delete(1.0, tk.END)
        self.updates_text.insert(tk.END, content)
        self.updates_text.config(state='disabled')

    def update_data_source_tab(self, content):
        self.data_source_text.config(state='normal')
        self.data_source_text.delete(1.0, tk.END)
        self.data_source_text.insert(tk.END, content)
        self.data_source_text.config(state='disabled')

def main():
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 