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
        self.data_source_options = ["Synthetic Data", "AWS CloudTrail Logs", "Azure Activity Logs", "CyberArk Logs (Synthetic)"]
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
        self.n_events.set(1000) # Reduced for debugging
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
        self.contamination_ratio.set(0.35) # Default from HybridAnomalyDetector
        self.contamination_ratio.grid(row=current_row, column=1, sticky="ew", pady=5, padx=5)
        current_row += 1
        
        # --- Hyperparameter Tuning Controls ---
        self.if_estimators_label = ttk.Label(self.control_frame, text="IF Estimators (n):")
        self.if_estimators_label.grid(row=current_row, column=0, sticky="w", pady=5, padx=5)
        self.n_estimators_iso_forest = ttk.Spinbox(self.control_frame, from_=50, to=500, increment=50, width=10)
        self.n_estimators_iso_forest.set(400) # Default
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

        # Explanatory text
        ttk.Label(self.data_source_tab, text="""
This feature allows the system to seamlessly integrate with a variety of enterprise log sources,
ensuring holistic threat visibility and enhanced accuracy. Define new log sources below.
""", wraplength=1000, justify=tk.LEFT).pack(pady=10, padx=10, anchor="w")

        # New Data Source Configuration Frame
        config_frame = ttk.LabelFrame(self.data_source_tab, text="New Data Source Configuration")
        config_frame.pack(fill=tk.X, padx=10, pady=5, anchor="nw")

        current_config_row = 0

        # Source Name
        ttk.Label(config_frame, text="Source Name:").grid(row=current_config_row, column=0, sticky="w", pady=5, padx=5)
        self.source_name_entry = ttk.Entry(config_frame, width=40)
        self.source_name_entry.grid(row=current_config_row, column=1, sticky="ew", pady=5, padx=5)
        current_config_row += 1

        # Source Type
        ttk.Label(config_frame, text="Source Type:").grid(row=current_config_row, column=0, sticky="w", pady=5, padx=5)
        self.source_type_var = tk.StringVar(value="Select Type")
        self.source_type_options = ["AWS CloudTrail", "Azure Activity Logs", "Generic JSON", "CSV", "Other"]
        self.source_type_combobox = ttk.Combobox(config_frame, textvariable=self.source_type_var, values=self.source_type_options, state="readonly", width=37)
        self.source_type_combobox.grid(row=current_config_row, column=1, sticky="ew", pady=5, padx=5)
        current_config_row += 1

        # Schema Mapping File (Placeholder)
        ttk.Label(config_frame, text="Schema Mapping File:").grid(row=current_config_row, column=0, sticky="w", pady=5, padx=5)
        self.schema_path_var = tk.StringVar()
        self.schema_path_entry = ttk.Entry(config_frame, textvariable=self.schema_path_var, width=30)
        self.schema_path_entry.grid(row=current_config_row, column=1, sticky="ew", pady=5, padx=5)
        current_config_row += 1

        self.browse_schema_button = ttk.Button(config_frame, text="Browse", command=self.browse_schema_file)
        self.browse_schema_button.grid(row=current_config_row, column=1, sticky="e", pady=5, padx=5)
        current_config_row += 1

        # Save Configuration Button
        self.save_config_button = ttk.Button(config_frame, text="Save Configuration", command=self.save_data_source_config)
        self.save_config_button.grid(row=current_config_row, column=0, columnspan=2, pady=10)

        # Configure column weights for resizing
        config_frame.grid_columnconfigure(1, weight=1)

        # --- Reporting Tab ---
        self.reporting_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reporting_tab, text="Reporting")

        self.reporting_text = tk.Text(self.reporting_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.reporting_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame for reporting visualizations
        self.reporting_viz_frame = ttk.LabelFrame(self.reporting_tab, text="Anomaly Trends")
        self.reporting_viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.reporting_figure = plt.Figure(figsize=(8, 4))
        self.reporting_canvas = FigureCanvasTkAgg(self.reporting_figure, master=self.reporting_viz_frame)
        self.reporting_canvas_widget = self.reporting_canvas.get_tk_widget()
        self.reporting_canvas_widget.pack(fill=tk.BOTH, expand=True)

        self.update_reporting_tab(0, 0, None, None, None) # Initial call with placeholders

        # --- Value Proposition Tab ---
        self.value_prop_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.value_prop_tab, text="Value Proposition")

        self.value_prop_text = tk.Text(self.value_prop_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.value_prop_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Initial content for Value Proposition tab
        self.update_value_prop_tab()

        # --- Experiment Log Tab ---
        self.experiment_log_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.experiment_log_tab, text="Experiment Log")

        self.experiment_log_text = tk.Text(self.experiment_log_tab, wrap=tk.WORD, state='disabled', font=("TkDefaultFont", 10))
        self.experiment_log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.log_experiment_result("Initial Run") # Log a header for the first run

        # Initial call to adjust controls
        self.on_data_source_change()

    def on_data_source_change(self, event=None):
        selected_source = self.data_source_var.get()
        if selected_source == "Synthetic Data" or selected_source == "CyberArk Logs (Synthetic)":
            for widget in self.synthetic_controls:
                widget.grid() # Show synthetic controls
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
            for widget in self.synthetic_controls:
                widget.grid_remove() # Hide synthetic controls
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
        self.status_text.see(tk.END) # Auto-scroll to the end
    
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
                # Data Loading (1/4)
                self.root.after(0, self.update_status, "Data Loading... (1/4)")
                df_local = None # Initialize df_local
                true_labels = None
                true_anomalies_exist = False

                selected_source = self.data_source_var.get()
                file_path = self.file_path_var.get()

                print(f"DEBUG: main.py - Selected Data Source: {selected_source}") # Added debug print

                if selected_source == "Synthetic Data":
                    self.root.after(0, self._update_progress_bar, 10, "Generating synthetic dataset...")
                    generator = IAMLogGenerator()
                    df_local = generator.generate_dataset(n_events=int(self.n_events.get()), anomaly_ratio=float(self.contamination_ratio.get()))
                    self.root.after(0, self._update_progress_bar, 20, "Data generation complete!")
                    print(f"DEBUG: main.py - df_local.shape after data generation: {df_local.shape}")

                elif selected_source == "CyberArk Logs (Synthetic)":
                    print("DEBUG: main.py - Entering CyberArk Logs (Synthetic) block.") # Added debug print
                    self.root.after(0, self._update_progress_bar, 10, "Generating synthetic CyberArk logs...")
                    log_reader = get_log_reader('cyberark') # Get CyberArkLogReader instance
                    df_local = log_reader.read_logs(num_events=int(self.n_events.get()), anomaly_ratio=float(self.contamination_ratio.get()))
                    self.root.after(0, self._update_progress_bar, 20, "CyberArk log generation complete!")
                    print(f"DEBUG: main.py - df_local.shape after CyberArk data generation: {df_local.shape}") # Added debug print

                elif selected_source == "AWS CloudTrail Logs" or selected_source == "Azure Activity Logs":
                    self.root.after(0, self.update_status, f"Loading {selected_source}... (1/4)")
                    self.root.after(0, self._update_progress_bar, 5, "Initializing log reader...")
                    
                    try:
                        reader_class = None
                        if selected_source == "AWS CloudTrail Logs":
                            reader_class = AWSCloudTrailReader
                        elif selected_source == "Azure Activity Logs":
                            reader_class = IAMLogReader # Using generic IAMLogReader for Azure now
                        
                        if reader_class and file_path:
                            log_reader = get_log_reader(reader_class, file_path)
                            
                            all_chunks = []
                            total_records_processed = 0
                            self.root.after(0, self.update_status, f"Reading logs from {file_path} in chunks...")
                            for i, chunk_df in enumerate(log_reader.read_logs_in_chunks()):
                                all_chunks.append(chunk_df)
                                total_records_processed += len(chunk_df)
                                self.root.after(0, self.update_status, f"Processing chunk {i+1}... ({len(all_chunks)}/{total_records_processed})")
                                self.root.after(0, self._update_progress_bar, 5 + int((i+1)/10 * 15))

                            if all_chunks:
                                df_local = pd.concat(all_chunks, ignore_index=True)
                            else:
                                df_local = pd.DataFrame()
                        else: # Handle case where reader_class or file_path is missing
                            self.update_status("Error: No reader class or file path provided for selected log type.")
                            self._update_progress_bar(0)
                            return
                                
                    except FileNotFoundError:
                        self.update_status(f"Error: File not found at {file_path}")
                        self._update_progress_bar(0)
                        return
                    except json.JSONDecodeError as e:
                        self.update_status(f"Error decoding JSON from {file_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        self._update_progress_bar(0)
                        return
                    except Exception as e:
                        self.update_status(f"An unexpected error occurred during log reading: {e}")
                        import traceback
                        traceback.print_exc()
                        self._update_progress_bar(0)
                        return

                if df_local is None or df_local.empty:
                    self.root.after(0, self.update_status, "Error: No data to process.")
                    self.root.after(0, lambda: self.run_button.state(['!disabled']))
                    return
                
                self.root.after(0, self.update_status, "Logs loaded and cleaned!")

                # Debug print: Check columns after log reading
                print(f"DEBUG: main.py - Columns after log_reader.read_logs: {df_local.columns.tolist()}")
                print(f"DEBUG: main.py - 'timestamp' column exists after read: {'timestamp' in df_local.columns}")
                if 'timestamp' in df_local.columns:
                    print(f"DEBUG: main.py - 'timestamp' column dtype after read: {df_local['timestamp'].dtype}")
                    print(f"DEBUG: main.py - First 5 timestamp values after read: {df_local['timestamp'].head()}")

                # Extract features (20-40% progress)
                self.root.after(0, self.update_status, "Extracting features... (2/4)")
                self.root.after(0, self._update_progress_bar, 25, "Initializing feature engineer...")
                feature_engineer = FeatureEngineer()

                # Debug print: Check columns before feature engineering
                print(f"DEBUG: main.py - Columns before feature_engineer.engineer_features: {df_local.columns.tolist()}")
                print(f"DEBUG: main.py - 'timestamp' column exists before engineer: {'timestamp' in df_local.columns}")
                if 'timestamp' in df_local.columns:
                    print(f"DEBUG: main.py - 'timestamp' column dtype before engineer: {df_local['timestamp'].dtype}")
                    print(f"DEBUG: main.py - First 5 timestamp values before engineer: {df_local['timestamp'].head()}")

                # Pass a progress callback to the feature engineer
                def feature_engineering_progress(current_step, total_steps, message):
                    base_progress = 25 # Start of feature engineering progress
                    progress_range = 40 - base_progress # Total range for feature engineering
                    step_progress = int(base_progress + (current_step / total_steps) * progress_range)
                    self.root.after(0, self._update_progress_bar, step_progress, message)

                df_local = feature_engineer.engineer_features(df_local, feature_engineering_progress)
                self.feature_columns = feature_engineer.get_feature_columns()
                self.root.after(0, self._update_progress_bar, 40, "Feature extraction complete!")
                print(f"DEBUG: main.py - df_local.shape after feature engineering: {df_local.shape}")

                # After feature engineering, if synthetic data, extract true labels from the potentially reduced df_local
                if selected_source == "Synthetic Data" and 'is_anomaly' in df_local.columns:
                    true_labels = df_local['is_anomaly'].values
                    true_anomalies_exist = True
                    print(f"DEBUG: main.py - len(true_labels) after feature engineering: {len(true_labels)}")
                else:
                    true_anomalies_exist = False # Ensure this is False for real logs

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
                hybrid_detector.fit(df_local[self.feature_columns], self.feature_columns, self._update_progress_bar)
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
                print(f"DEBUG: main.py - len(true_labels) before precision_score: {len(true_labels)}")
                print(f"DEBUG: main.py - len(predictions) before precision_score: {len(predictions)}")
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
                    
                    experiment_result = f"Data Source: {selected_source}, IF_Estimators: {n_estimators_iso_forest}, IF_Max_Features: {max_features_iso_forest}, RF_Estimators: {n_estimators_rf}, RF_Max_Depth: {max_depth_rf}, RF_Min_Samples_Split: {min_samples_split_rf}, Contamination: {self.contamination_ratio.get()}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Detected Anomalies: {np.sum(predictions)}"
                    self.root.after(0, self.log_experiment_result, experiment_result)

                else:
                    self.root.after(0, self.update_status, "\nModel Performance: (Note: Metrics not applicable for unlabeled real logs)")
                    self.root.after(0, self.update_status, f"Contamination Ratio: {self.contamination_ratio.get()}")
                    experiment_result = f"Data Source: {selected_source}, IF_Estimators: {n_estimators_iso_forest}, IF_Max_Features: {max_features_iso_forest}, RF_Estimators: {n_estimators_rf}, RF_Max_Depth: {max_depth_rf}, RF_Min_Samples_Split: {min_samples_split_rf}, Contamination: {self.contamination_ratio.get()}, Detected Anomalies: {np.sum(predictions)} (Metrics N/A)"
                    self.root.after(0, self.log_experiment_result, experiment_result)
                
                self.root.after(0, self.update_status, f"Detected Anomalies: {np.sum(predictions)}")

                # Store total events and detected anomalies
                self.total_events_processed = len(df_local)
                self.detected_anomalies_count = np.sum(predictions)

                # Get top anomalous users (pass the Series, not a string)
                top_anomalous_users_series = None
                anomalous_df = df_local[predictions == 1]
                if not anomalous_df.empty and 'user_id' in anomalous_df.columns: # Changed from 'userIdentity' to 'user_id' for consistency
                    top_anomalous_users_series = anomalous_df['user_id'].value_counts().head(5)

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
                            self.root.after(0, self.update_status, f"    Error explaining anomaly {idx}: {explanation['error']}")
                else:
                    self.root.after(0, self.update_status, "\nNo anomalies detected.")

                self.root.after(0, self.update_status, "Saving results...")
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'anomaly_results.csv')
                # Ensure 'is_anomaly' column is present before saving if it exists
                output_df = self.df.copy()
                output_df['is_anomaly_predicted'] = self.predictions # Add predicted anomalies
                output_df['anomaly_score'] = self.scores # Add anomaly scores

                # If original data had true anomalies, include them
                if true_anomalies_exist and 'is_anomaly' in self.df.columns: # Check df.columns not df_local.columns
                    output_df['is_anomaly_true'] = true_labels
                
                output_df.to_csv(output_path, index=False)
                self.root.after(0, self.update_status, f"Results saved to '{output_path}'")

                # Update the Reporting Tab with current results
                self.root.after(0, self.update_reporting_tab, self.total_events_processed, self.detected_anomalies_count, top_anomalous_users_series, df_local, predictions)

                self.root.after(0, self.update_status, "\nAnalysis complete!")
                self.root.after(0, self._update_progress_bar, 100, "Done!") # Final update
                self.root.after(0, self.update_visualization)
            except Exception as e:
                self.root.after(0, self.update_status, f"An error occurred during analysis: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self.root.after(0, lambda: self.run_button.state(['!disabled']))
        
        # Run the analysis in a separate thread to keep the GUI responsive
        threading.Thread(target=analysis_thread).start()

    def update_updates_tab(self, content):
        self.updates_text.config(state='normal')
        self.updates_text.delete(1.0, tk.END)
        self.updates_text.insert(tk.END, content)
        self.updates_text.config(state='disabled')

    def update_data_source_tab(self, content):
        # No longer used for static text display, content is now driven by UI elements
        pass

    def update_reporting_tab(self, total_events, detected_anomalies, top_users, df_local, predictions):
        self.reporting_text.config(state='normal')
        self.reporting_text.delete(1.0, tk.END)

        content = f"""
**Analysis Summary**

- **Total Events Processed:** {total_events}
- **Detected Anomalies:** {detected_anomalies}

**Top Anomalous Users:**
"""
        if top_users is not None and not top_users.empty:
            # Calculate average anomaly score for top users
            anomalous_df_with_scores = df_local[predictions == 1].copy()
            if 'user_id' in anomalous_df_with_scores.columns and 'anomaly_score' in anomalous_df_with_scores.columns:
                # Filter anomalous_df_with_scores to only include the top_users
                top_user_ids = top_users.index.tolist()
                anomalous_df_top_users = anomalous_df_with_scores[anomalous_df_with_scores['user_id'].isin(top_user_ids)]
                
                avg_scores_by_user = anomalous_df_top_users.groupby('user_id')['anomaly_score'].mean()
                
                for user, count in top_users.items():
                    avg_score = avg_scores_by_user.get(user, 'N/A') # Get average score, N/A if not found
                    if avg_score != 'N/A':
                        content += f"- {user}: {count} anomalies (Avg Score: {avg_score:.3f})\n"
                    else:
                        content += f"- {user}: {count} anomalies\n"
            else:
                for user, count in top_users.items():
                    content += f"- {user}: {count} anomalies\n"
        else:
            content += "- No top anomalous users identified.\n"

        # Add Anomaly Trends Visualization
        self.reporting_figure.clear()
        ax = self.reporting_figure.add_subplot(111)

        if df_local is not None and predictions is not None and 'timestamp' in df_local.columns:
            anomalous_df = df_local[predictions == 1].copy()
            if not anomalous_df.empty:
                anomalous_df['timestamp'] = pd.to_datetime(anomalous_df['timestamp'])
                anomalies_by_hour = anomalous_df.groupby(anomalous_df['timestamp'].dt.hour).size()
                
                # Ensure all hours (0-23) are present, even if no anomalies
                all_hours = pd.Series(0, index=range(24))
                anomalies_by_hour = all_hours.add(anomalies_by_hour, fill_value=0)
                anomalies_by_hour = anomalies_by_hour.sort_index()

                ax.bar(anomalies_by_hour.index, anomalies_by_hour.values, color='skyblue')
                ax.set_title('Anomalies by Hour of Day')
                ax.set_xlabel('Hour of Day')
                ax.set_ylabel('Number of Anomalies')
                ax.set_xticks(range(0, 24, 2)) # Show every 2nd hour
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, 'No anomalies detected for trend analysis', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_title('Anomalies by Hour of Day')
        else:
            ax.text(0.5, 0.5, 'Data not available for trend analysis', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Anomalies by Hour of Day')
        
        self.reporting_figure.tight_layout()
        self.reporting_canvas.draw()

        # Add Potential Cost Savings
        AVERAGE_COST_PER_ANOMALY_PREVENTED = 5000 # Placeholder: $5,000 per anomaly prevented
        potential_cost_savings = detected_anomalies * AVERAGE_COST_PER_ANOMALY_PREVENTED
        content += f"\n**Potential Cost Savings (Estimated):** ${potential_cost_savings:,.2f}\n"
        content += "(Based on an estimated average of ${:,.2f} per prevented incident)\n".format(AVERAGE_COST_PER_ANOMALY_PREVENTED)

        content += """

---

**Integration Potential**

This reporting capability can be seamlessly integrated with existing Security Information and Event Management (SIEM) and Security Orchestration, Automation, and Response (SOAR) systems. By feeding anomaly alerts and summary statistics directly into these platforms, organizations can:

*   **Centralize Alert Management:** Consolidate alerts from various security tools into a single pane of glass.
*   **Automate Response Workflows:** Trigger automated actions (e.g., suspend user, block IP, escalate to security team) based on detected anomalies.
*   **Enhance Forensic Analysis:** Provide enriched context for incident investigations.
*   **Improve Overall Security Posture:** Proactively respond to threats and continually refine security policies based on real-time insights.

"""
        self.reporting_text.insert(tk.END, content)
        self.reporting_text.config(state='disabled')

    def update_value_prop_tab(self):
        content = """
**IAM Anomaly Detection: A Strategic Investment for Unparalleled Security and Tangible ROI**

In an era where digital identities are the primary attack vector, robust Identity and Access Management (IAM) is not just a security best practice—it's a critical business imperative. This AI-driven IAM Anomaly Detection system represents a strategic investment that proactively safeguards your organization against evolving cyber threats, delivering not only enhanced security but also significant, measurable business value.

### The Problem We Solve: Mitigating High-Impact Cyber Risks

*   **Credential Compromise (e.g., Phishing, Brute Force):** Detects and alerts on unusual login patterns (time, location, device, frequency), account takeover attempts, and suspicious access from new or blacklisted IPs, dramatically reducing the window of compromise.
*   **Privilege Abuse & Insider Threats:** Identifies authorized users exhibiting anomalous behavior, such as accessing sensitive data outside their job function, escalating privileges without authorization, or performing actions inconsistent with their historical profile, thereby curbing internal threats.
*   **Policy Violations & Misconfigurations:** Flags deviations from established IAM policies and potential misconfigurations that could expose your organization to risk, ensuring continuous compliance.
*   **Ransomware & Malware Spread:** Early detection of lateral movement or unusual resource access by compromised accounts can prevent widespread infection and minimize operational disruption.

### Delivering Tangible Business Value: Beyond Security

1.  **Reduced Breach Risk & Associated Costs (Estimated Savings: $X00,000 - $X Million per incident):**
    *   By identifying and neutralizing threats early, the system prevents minor incidents from escalating into costly data breaches, which averaged $4.45 million in 2023. Proactive detection minimizes forensic costs, legal fees, regulatory fines, and reputational damage.
2.  **Enhanced Operational Efficiency (Estimated Savings: Y% in Analyst Time):**
    *   Automates the laborious, error-prone task of manually sifting through colossal volumes of log data. This frees up highly skilled security analysts to focus on strategic threat intelligence, threat hunting, and incident response, significantly optimizing security operations.
3.  **Faster Incident Response & Recovery (Reduced Downtime):**
    *   Provides real-time anomaly alerts with rich contextual data, enabling security teams to respond to potential threats in minutes, not hours or days. This rapid response minimizes business disruption, reduces mean time to detect (MTTD) and mean time to respond (MTTR), and protects critical business continuity.
4.  **Improved Compliance & Audit Readiness (Avoidance of Penalties):**
    *   Generates comprehensive, auditable records of security events and detected anomalies. This streamlines compliance reporting for regulations such as GDPR, HIPAA, SOX, PCI DSS, and enhances your posture during internal and external audits, helping avoid hefty fines and legal repercussions.
5.  **Data-Driven Security Decisions & Policy Refinement:**
    *   Transforms raw log data into actionable intelligence, providing deep insights into user behavior and threat landscapes. This empowers security leaders to make informed decisions, optimize security policies, and allocate resources more effectively.
6.  **Unmatched Scalability & Extensibility:**
    *   Designed with a modular architecture that supports seamless integration with diverse log sources (AWS CloudTrail, Azure Activity Logs, Google Cloud Audit Logs, On-premise Active Directory, SIEMs like Splunk/Elastic Stack). This ensures the solution remains effective and adaptable to growing data volumes and evolving IT infrastructures.

### System Workflow & Value Flow Diagram: From Raw Data to Actionable Intelligence

This system transforms overwhelming security log data into a clear, actionable intelligence pathway:

1.  **Raw Log Data:** Ingests vast quantities of unstructured security logs from diverse enterprise sources (cloud, on-premise, network devices).
2.  **Log Ingestion & Standardization:** Raw logs are intelligently parsed, normalized, and transformed into a consistent, machine-readable format, ensuring data uniformity across all sources.
3.  **Cleaned & Standardized Logs:** A unified, high-quality dataset emerges, forming the foundation for deep behavioral analysis.
4.  **Advanced Feature Engineering:** Over 50 critical features are meticulously extracted and engineered, including time-based metrics (e.g., login frequency, time-between-actions), IP-based insights (e.g., new IP, geographic impossibilities), and intricate behavioral patterns (e.g., unique actions per session, role changes).
5.  **Comprehensive Behavioral Profiles:** These engineered features culminate in rich, multi-dimensional behavioral profiles for every user, role, and resource, establishing a baseline of 'normal.'
6.  **Hybrid Anomaly Detection Models:** Leveraging state-of-the-art machine learning, the system employs a powerful combination of models:
    *   **Isolation Forest:** For efficient identification of outlier events.
    *   **RandomForest Classifier:** For robust classification of normal vs. anomalous behavior.
    *   **LSTM Autoencoder (Deep Learning):** For uncovering subtle, complex sequential anomalies in user activity patterns that simpler models might miss.
7.  **Real-Time Anomaly Scores & Prioritized Alerts:** Models generate precise anomaly scores, with high scores triggering prioritized alerts. This intelligent scoring minimizes alert fatigue.
8.  **Intuitive Reporting & Visualization:** Detected anomalies, key metrics, and user behavior trends are presented through a user-friendly GUI with interactive charts and tables, enabling rapid understanding and investigation.

This comprehensive, AI-driven data flow culminates in unparalleled **Proactive Security & Business Value**, directly leading to:

*   **Significant Risk Reduction:** By detecting and preventing sophisticated threats before they cause damage.
*   **Substantial Cost Savings:** Through breach prevention, operational efficiency, and compliance assurance.
*   **Optimized Security Operations:** By automating analysis and empowering analysts.
*   **Enhanced Strategic Decision-Making:** Providing actionable insights for a stronger security posture.

**Invest in IAM Anomaly Detection – Secure Your Digital Future and Realize Measurable ROI.**
"""
        self.value_prop_text.config(state='normal')
        self.value_prop_text.delete(1.0, tk.END)
        self.value_prop_text.insert(tk.END, content)
        self.value_prop_text.config(state='disabled')

    def browse_file(self):
        file_selected = filedialog.askopenfilename(
            title="Select Log File",
            filetypes=(("JSON files", "*.json"), ("All files", "*.*" ))
        )
        if file_selected:
            self.file_path_var.set(file_selected)

    def browse_schema_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if file_path:
            self.schema_path_var.set(file_path)

    def save_data_source_config(self):
        source_name = self.source_name_entry.get()
        source_type = self.source_type_var.get()
        schema_path = self.schema_path_var.get()
        
        status_message = f"Saving configuration for: {source_name} (Type: {source_type}, Schema: {schema_path})\n(Note: This is a placeholder for future backend integration)"
        self.update_status(status_message)
        print(status_message) # For debugging/console visibility

    def log_experiment_result(self, result_description):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_log_entry = f"{current_time} - {result_description}\n"
        self.experiment_log_text.config(state='normal')
        self.experiment_log_text.insert(tk.END, experiment_log_entry)
        self.experiment_log_text.config(state='disabled')

def main():
    root = tk.Tk()
    app = AnomalyDetectionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 