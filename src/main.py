import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Tuple, List
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import json

from data_generator import IAMLogGenerator
from feature_engineering import FeatureEngineer
from models.hybrid_model import HybridAnomalyDetector
from data.iam_log_reader import get_log_reader, AWSCloudTrailReader, IAMLogReader # Import necessary reader classes

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
        self.data_source_options = ["Synthetic Data", "AWS CloudTrail Logs", "Azure Activity Logs"]
        self.data_source_combobox = ttk.Combobox(self.control_frame, textvariable=self.data_source_var, values=self.data_source_options, state="readonly")
        self.data_source_combobox.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        self.data_source_combobox.bind("<<ComboboxSelected>>", self.on_data_source_change)
        current_row += 1

        # File path selection for real logs
        self.file_path_var = tk.StringVar()
        self.file_path_label = ttk.Label(self.control_frame, text="Log File Path:")
        self.file_path_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.file_path_entry = ttk.Entry(self.control_frame, textvariable=self.file_path_var, width=30)
        self.file_path_entry.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1

        self.browse_button = ttk.Button(self.control_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=current_row, column=1, sticky="e", pady=5, padx=5)
        current_row += 1

        # Synthetic Data Controls
        self.num_events_label = ttk.Label(self.control_frame, text="Number of Events:")
        self.num_events_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_events = ttk.Spinbox(self.control_frame, from_=100, to=5000, width=10)
        self.n_events.set(500)
        self.n_events.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        self.num_users_label = ttk.Label(self.control_frame, text="Number of Users:")
        self.num_users_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_users = ttk.Spinbox(self.control_frame, from_=10, to=200, width=10)
        self.n_users.set(20)
        self.n_users.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        self.num_roles_label = ttk.Label(self.control_frame, text="Number of Roles:")
        self.num_roles_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_roles = ttk.Spinbox(self.control_frame, from_=2, to=20, width=10)
        self.n_roles.set(3)
        self.n_roles.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        self.num_actions_label = ttk.Label(self.control_frame, text="Number of Actions:")
        self.num_actions_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_actions = ttk.Spinbox(self.control_frame, from_=5, to=50, width=10)
        self.n_actions.set(5)
        self.n_actions.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        self.contamination_ratio_label = ttk.Label(self.control_frame, text="Contamination Ratio:")
        self.contamination_ratio_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.contamination_ratio = ttk.Spinbox(self.control_frame, from_=0.01, to=0.5, increment=0.01, width=10, format="%.2f")
        self.contamination_ratio.set(0.15) # Default from HybridAnomalyDetector
        self.contamination_ratio.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        # --- Hyperparameter Tuning Controls ---
        self.if_estimators_label = ttk.Label(self.control_frame, text="IF Estimators (n):")
        self.if_estimators_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_estimators_iso_forest = ttk.Spinbox(self.control_frame, from_=50, to=500, increment=50, width=10)
        self.n_estimators_iso_forest.set(300) # Default
        self.n_estimators_iso_forest.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1

        self.if_max_features_label = ttk.Label(self.control_frame, text="IF Max Features (float):")
        self.if_max_features_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.max_features_iso_forest = ttk.Spinbox(self.control_frame, from_=0.1, to=1.0, increment=0.1, width=10, format="%.1f")
        self.max_features_iso_forest.set(1.0) # Default
        self.max_features_iso_forest.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1

        self.rf_estimators_label = ttk.Label(self.control_frame, text="RF Estimators (n):")
        self.rf_estimators_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_estimators_rf = ttk.Spinbox(self.control_frame, from_=50, to=500, increment=50, width=10)
        self.n_estimators_rf.set(250) # Default
        self.n_estimators_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1

        self.rf_max_depth_label = ttk.Label(self.control_frame, text="RF Max Depth (int/None):")
        self.rf_max_depth_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        # Using Combobox for Max Depth to allow 'None'
        self.max_depth_rf_var = tk.StringVar(value="30")
        self.max_depth_rf = ttk.Combobox(self.control_frame, textvariable=self.max_depth_rf_var, values=["None", 10, 20, 30, 50], state="readonly", width=10)
        self.max_depth_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1

        self.rf_min_samples_split_label = ttk.Label(self.control_frame, text="RF Min Samples Split (n):")
        self.rf_min_samples_split_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.min_samples_split_rf = ttk.Spinbox(self.control_frame, from_=2, to=20, width=10)
        self.min_samples_split_rf.set(2) # Default
        self.min_samples_split_rf.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        # --- End Hyperparameter Tuning Controls ---
        
        # List of widgets for synthetic data controls
        self.synthetic_controls = [
            self.num_events_label, self.n_events,
            self.num_users_label, self.n_users,
            self.num_roles_label, self.n_roles,
            self.num_actions_label, self.n_actions,
            self.contamination_ratio_label, self.contamination_ratio
        ]

        # List of widgets for model tuning controls
        self.model_tuning_controls = [
            self.if_estimators_label, self.n_estimators_iso_forest,
            self.if_max_features_label, self.max_features_iso_forest,
            self.rf_estimators_label, self.n_estimators_rf,
            self.rf_max_depth_label, self.max_depth_rf,
            self.rf_min_samples_split_label, self.min_samples_split_rf
        ]

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

        # --- Reporting Tab ---
        self.reporting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reporting_tab, text="Reporting")

        self.reporting_text = tk.Text(self.reporting_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.reporting_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.update_reporting_tab("""
**Anomaly Reporting and Integration Capabilities**

This section highlights the reporting and integration capabilities of the IAM Anomaly Detection system. While the current version focuses on core detection, future enhancements will provide robust reporting and seamless integration with existing security ecosystems.

**Last Analysis Summary:**
*   **Total Events Processed:** [N/A - Run analysis to populate]
*   **Detected Anomalies:** [N/A - Run analysis to populate]
*   **Top Anomalous Users:** [N/A - Run analysis to populate]

**Key Reporting Features (Planned):**
*   **Customizable Dashboards:** Interactive dashboards to visualize anomaly trends, user behavior, and security incidents over time.
*   **Detailed Anomaly Reports:** Generate comprehensive reports for individual anomalies, including contextual information, contributing features, and severity levels.
*   **Scheduled Reporting:** Automate the generation and distribution of daily, weekly, or monthly anomaly summaries.

**Integration with Security Ecosystems (Planned):**
*   **SIEM Integration (e.g., Splunk, Elastic Security, Microsoft Sentinel):** Push detected anomalies and their context directly to SIEM platforms for centralized logging, correlation, and alert management.
*   **SOAR Integration (e.g., Palo Alto Networks XSOAR, Splunk SOAR):** Trigger automated incident response playbooks based on high-severity anomalies, enabling rapid containment and remediation.
*   **API for Custom Integrations:** Provide a robust API for third-party tools and custom scripts to query anomaly data and integrate with other enterprise systems.

These planned features underscore the system's potential to become a pivotal component in an organization's overall security posture, enabling proactive threat hunting and streamlined incident response.
""")

        # Initial call to adjust controls
        self.on_data_source_change()

    def on_data_source_change(self, event=None):
        selected_source = self.data_source_var.get()
        if selected_source == "Synthetic Data":
            # Show synthetic controls
            for widget in self.synthetic_controls:
                widget.grid()
            self.n_events.config(state="enabled")
            self.n_users.config(state="enabled")
            self.n_roles.config(state="enabled")
            self.n_actions.config(state="enabled")
            self.contamination_ratio.config(state="enabled")

            # Show model tuning controls (always visible for synthetic data)
            for widget in self.model_tuning_controls:
                widget.grid()
            self.n_estimators_iso_forest.config(state="enabled")
            self.max_features_iso_forest.config(state="enabled")
            self.n_estimators_rf.config(state="enabled")
            self.max_depth_rf.config(state="readonly")
            self.min_samples_split_rf.config(state="enabled")

            # Hide file path selection
            self.file_path_label.grid_remove()
            self.file_path_entry.grid_remove()
            self.browse_button.grid_remove()

        else: # Real Logs (AWS CloudTrail Logs or Azure Activity Logs) selected
            # Hide synthetic controls
            for widget in self.synthetic_controls:
                widget.grid_remove()
            self.n_events.config(state="disabled")
            self.n_users.config(state="disabled")
            self.n_roles.config(state="disabled")
            self.n_actions.config(state="disabled")
            self.contamination_ratio.config(state="disabled")

            # Show model tuning controls (always visible for real logs too)
            for widget in self.model_tuning_controls:
                widget.grid()
            self.n_estimators_iso_forest.config(state="enabled")
            self.max_features_iso_forest.config(state="enabled")
            self.n_estimators_rf.config(state="enabled")
            self.max_depth_rf.config(state="readonly")
            self.min_samples_split_rf.config(state="enabled")

            # Show file path selection
            self.file_path_label.grid()
            self.file_path_entry.grid()
            self.browse_button.grid()
    
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
                elif selected_source == "AWS CloudTrail Logs" or selected_source == "Azure Activity Logs":
                    self.root.after(0, self.update_status, f"Loading {selected_source}... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing log reader...")
                    reader = IAMLogReader()
                    file_path = self.file_path_var.get()

                    if not file_path:
                        self.update_status("Error: Please select a log file path.")
                        self._update_progress_bar(0)
                        return

                    self.root.after(0, self.update_status, f"Reading logs from {file_path} in chunks...")
                    all_chunks = []
                    chunk_size = 500 # Define a suitable chunk size for processing
                    
                    try:
                        with open(file_path, 'r') as f:
                            # For JSON, we need to read the whole file first or handle streaming JSON (more complex)
                            # For simplicity, if we expect very large files that can't fit in memory, 
                            # a different reading strategy (e.g., line-by-line for JSONL, or a custom parser)
                            # would be needed. For now, we load fully then process for chunking effect.
                            raw_data = json.load(f)
                            if selected_source == "AWS CloudTrail Logs":
                                records = raw_data.get('Records', [])
                            else: # Azure Activity Logs
                                records = raw_data # Azure Activity Logs are typically a list of dicts at top level

                        total_records = len(records)
                        self.root.after(0, self.update_status, f"Total records to process: {total_records}")

                        for i in range(0, total_records, chunk_size):
                            chunk_records = records[i:i + chunk_size]
                            if selected_source == "AWS CloudTrail Logs":
                                # Create a temporary structure that mimics the original file structure for the reader
                                temp_data = {'Records': chunk_records}
                                chunk_df = reader.read_aws_cloudtrail_logs_from_data(temp_data) # New method
                            else: # Azure Activity Logs
                                chunk_df = reader.read_azure_activity_logs_from_data(chunk_records) # New method
                            
                            if not chunk_df.empty:
                                all_chunks.append(chunk_df)
                            
                            progress = 5 + int(((i + chunk_size) / total_records) * (20 - 5)) # Scale progress from 5% to 20%
                            self.root.after(0, self._update_progress_bar, progress, f"Processing chunk {i//chunk_size + 1}... ({min(i + chunk_size, total_records)}/{total_records})")
                        
                        if all_chunks:
                            df_local = pd.concat(all_chunks, ignore_index=True)
                        else:
                            df_local = pd.DataFrame()
                            
                    except FileNotFoundError:
                        self.update_status(f"Error: File not found at {file_path}")
                        self._update_progress_bar(0)
                        return
                    except json.JSONDecodeError as e:
                        self.update_status(f"Error decoding JSON from {file_path}: {e}")
                        self._update_progress_bar(0)
                        return
                    except Exception as e:
                        self.update_status(f"An unexpected error occurred during log reading: {e}")
                        import traceback
                        traceback.print_exc()
                        self._update_progress_bar(0)
                        return

                    self.root.after(0, self.update_status, "Logs loaded and cleaned!")
                    true_anomalies_exist = False # No true labels for real logs
                
                if df_local is None or df_local.empty:
                    self.root.after(0, self.update_status, "Error: No data to process.")
                    self.root.after(0, lambda: self.run_button.state(['!disabled']))
                    return
                
                # Extract features (20-40% progress)
                self.root.after(0, self.update_status, "Extracting features... (2/4)")
                self.root.after(0, self._update_progress_bar, 25, "Initializing feature engineer...")
                feature_engineer = FeatureEngineer()

                # Pass a progress callback to the feature engineer
                def feature_engineering_progress(current_step, total_steps, message):
                    base_progress = 25 # Start of feature engineering progress
                    progress_range = 40 - base_progress # Total range for feature engineering
                    step_progress = int(base_progress + (current_step / total_steps) * progress_range)
                    self.root.after(0, self._update_progress_bar, step_progress, message)

                df_local = feature_engineer.engineer_features(df_local, progress_callback=feature_engineering_progress)
                feature_columns = feature_engineer.get_feature_columns()
                self.root.after(0, self._update_progress_bar, 40, "Feature extraction complete!")
                
                # Train model (40-85% progress)
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

                # Adjust progress range for model training within hybrid_model.py
                # hybrid_detector.fit will update progress from 45% to 85%
                hybrid_detector.fit(df_local[feature_columns], feature_columns, self._update_progress_bar) 
                self.root.after(0, self._update_progress_bar, 85, "Model trained.") # Adjusted percentage

                # Making predictions (85-95% progress)
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

                # Anomaly Explanation
                anomalous_indices = np.where(predictions == 1)[0]
                if len(anomalous_indices) > 0:
                    self.root.after(0, self.update_status, "\nAnomaly Explanations (Top 5 features for first 3 anomalies):")
                    for i, idx in enumerate(anomalous_indices):
                        if i >= 3: # Limit to first 3 anomalies for explanation in GUI
                            break
                        anomaly_data_row = df_local.iloc[[idx]]
                        explanation = hybrid_detector.explain_anomaly(anomaly_data_row) # Pass the single row DataFrame
                        
                        if "error" not in explanation:
                            self.root.after(0, self.update_status, f"  Anomaly {i+1} (Original Index: {idx}):")
                            top_features = list(explanation.items())[:5] # Get top 5 features
                            for feature, importance in top_features:
                                self.root.after(0, self.update_status, f"    - {feature}: {importance:.4f}")
                        else:
                            error_message = explanation['error']
                            self.root.after(0, self.update_status, f"  Anomaly {i+1} (Original Index: {idx}): {error_message}")

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

                # Update Reporting Tab with summary
                total_events = len(df_local) # Assuming df_local contains all processed events
                detected_anomalies_count = np.sum(predictions)
                
                top_anomalous_users_str = "N/A"
                if 'user_id' in df_local.columns and not df_local['user_id'].empty:
                    anomaly_users_counts = df_local[predictions == 1]['user_id'].value_counts()
                    if not anomaly_users_counts.empty:
                        top_anomalous_users = anomaly_users_counts.head(5).index.tolist()
                        top_anomalous_users_str = ", ".join(top_anomalous_users)
                    
                reporting_content = f"""
**Anomaly Reporting and Integration Capabilities**

This section highlights the reporting and integration capabilities of the IAM Anomaly Detection system. While the current version focuses on core detection, future enhancements will provide robust reporting and seamless integration with existing security ecosystems.

**Last Analysis Summary:**
*   **Total Events Processed:** {total_events}
*   **Detected Anomalies:** {detected_anomalies_count}
*   **Top Anomalous Users:** {top_anomalous_users_str}

**Key Reporting Features (Planned):**
*   **Customizable Dashboards:** Interactive dashboards to visualize anomaly trends, user behavior, and security incidents over time.
*   **Detailed Anomaly Reports:** Generate comprehensive reports for individual anomalies, including contextual information, contributing features, and severity levels.
*   **Scheduled Reporting:** Automate the generation and distribution of daily, weekly, or monthly anomaly summaries.

**Integration with Security Ecosystems (Planned):**
*   **SIEM Integration (e.g., Splunk, Elastic Security, Microsoft Sentinel):** Push detected anomalies and their context directly to SIEM platforms for centralized logging, correlation, and alert management.
*   **SOAR Integration (e.g., Palo Alto Networks XSOAR, Splunk SOAR):** Trigger automated incident response playbooks based on high-severity anomalies, enabling rapid containment and remediation.
*   **API for Custom Integrations:** Provide a robust API for third-party tools and custom scripts to query anomaly data and integrate with other enterprise systems.

These planned features underscore the system's potential to become a pivotal component in an organization's overall security posture, enabling proactive threat hunting and streamlined incident response.
"""
                self.root.after(0, self.update_reporting_tab, reporting_content)

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

    def update_reporting_tab(self, content):
        self.reporting_text.config(state='normal')
        self.reporting_text.delete(1.0, tk.END)
        self.reporting_text.insert(tk.END, content)
        self.reporting_text.config(state='disabled')

    def browse_file(self):
        file_selected = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*" ))
        )
        if file_selected:
            self.file_path_var.set(file_selected)

def main():
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 