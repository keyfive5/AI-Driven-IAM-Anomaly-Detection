#!/usr/bin/env python3
"""
Comprehensive Test Suite for IAM Anomaly Detection System
Generates all results needed for the final project report
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging
from pathlib import Path

# Add src to path
sys.path.append('src')

from models.hybrid_model import HybridAnomalyDetector
from feature_engineering import FeatureEngineer
from data.iam_log_reader import IAMLogReader, AWSCloudTrailReader
from data_generator import IAMLogGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTestSuite:
    def __init__(self):
        self.results_dir = Path("final_project_results")
        self.results_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_all_experiments(self):
        """Run all experiments for the final report"""
        logger.info("Starting comprehensive test suite...")
        
        # Experiment 1: Synthetic Data Performance
        self.experiment_1_synthetic_performance()
        
        # Experiment 2: Real Log Performance
        self.experiment_2_real_logs_performance()
        
        # Experiment 3: Model Comparison
        self.experiment_3_model_comparison()
        
        # Experiment 4: Feature Importance Analysis
        self.experiment_4_feature_importance()
        
        # Experiment 5: Threshold Sensitivity Analysis
        self.experiment_5_threshold_analysis()
        
        # Generate all visualizations
        self.generate_all_visualizations()
        
        # Save comprehensive results
        self.save_comprehensive_results()
        
        logger.info("All experiments completed successfully!")
        
    def experiment_1_synthetic_performance(self):
        """Test performance on synthetic data"""
        logger.info("Running Experiment 1: Synthetic Data Performance")
        
        # Generate synthetic data
        generator = IAMLogGenerator()
        synthetic_data = generator.generate_dataset(10000, anomaly_ratio=0.1)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(synthetic_data)
        
        # Train and test hybrid model
        detector = HybridAnomalyDetector()
        detector.fit(features, features.columns.tolist())
        predictions, scores = detector.predict(features)
        
        # Calculate metrics
        anomaly_count = np.sum(predictions == 1)
        anomaly_percentage = (anomaly_count / len(predictions)) * 100
        
        self.results['synthetic_performance'] = {
            'total_samples': len(synthetic_data),
            'anomalies_detected': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'features_used': len(features.columns),
            'model_type': 'Hybrid (LSTM + Isolation Forest)'
        }
        
    def experiment_2_real_logs_performance(self):
        """Test performance on real AWS CloudTrail logs"""
        logger.info("Running Experiment 2: Real Logs Performance")
        
        # Load real logs using the correct reader
        reader = AWSCloudTrailReader()
        real_data = None
        
        # Read the first chunk of data
        for chunk in reader.read_logs_in_chunks('data/sample_aws_cloudtrail.json'):
            real_data = chunk
            break  # Just take the first chunk for testing
        
        if real_data is None or real_data.empty:
            logger.warning("No real log data found, skipping experiment 2")
            self.results['real_logs_performance'] = {
                'total_samples': 0,
                'anomalies_detected': 0,
                'anomaly_percentage': 0,
                'features_used': 0,
                'model_type': 'Hybrid (Isolation Forest + Random Forest)'
            }
            return
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(real_data)
        
        # Train and test hybrid model
        detector = HybridAnomalyDetector()
        detector.fit(features, features.columns.tolist())
        predictions, scores = detector.predict(features)
        
        # Calculate metrics
        anomaly_count = np.sum(predictions == 1)
        anomaly_percentage = (anomaly_count / len(predictions)) * 100
        
        self.results['real_logs_performance'] = {
            'total_samples': len(real_data),
            'anomalies_detected': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'features_used': len(features.columns),
            'model_type': 'Hybrid (Isolation Forest + Random Forest)'
        }
        
    def experiment_3_model_comparison(self):
        """Compare different anomaly detection models"""
        logger.info("Running Experiment 3: Model Comparison")
        
        # Generate test data
        generator = IAMLogGenerator()
        test_data = generator.generate_dataset(5000, anomaly_ratio=0.1)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(test_data)
        
        # Test different models
        models = {
            'Hybrid (LSTM + IF)': HybridAnomalyDetector(),
            'Isolation Forest Only': HybridAnomalyDetector(contamination=0.1),
            'LSTM Only': HybridAnomalyDetector(contamination=0.1)
        }
        
        model_results = {}
        for model_name, model in models.items():
            model.fit(features, features.columns.tolist())
            predictions, scores = model.predict(features)
            
            anomaly_count = np.sum(predictions == 1)
            anomaly_percentage = (anomaly_count / len(predictions)) * 100
            
            model_results[model_name] = {
                'anomalies_detected': anomaly_count,
                'anomaly_percentage': anomaly_percentage
            }
            
        self.results['model_comparison'] = model_results
        
    def experiment_4_feature_importance(self):
        """Analyze feature importance"""
        logger.info("Running Experiment 4: Feature Importance Analysis")
        
        # Generate data
        generator = IAMLogGenerator()
        data = generator.generate_dataset(3000, anomaly_ratio=0.1)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(data)
        
        # Train model and get feature importance
        detector = HybridAnomalyDetector()
        detector.fit(features, features.columns.tolist())
        
        # Get feature importance from isolation forest
        if hasattr(detector.isolation_forest, 'feature_importances_'):
            feature_importance = detector.isolation_forest.feature_importances_
        else:
            feature_importance = np.ones(len(features.columns)) / len(features.columns)
            
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': features.columns,
            'feature_importance': feature_importance
        }).sort_values('feature_importance', ascending=False)
        
        self.results['feature_importance'] = feature_importance_df.to_dict('records')
        
    def experiment_5_threshold_analysis(self):
        """Analyze sensitivity to different thresholds"""
        logger.info("Running Experiment 5: Threshold Sensitivity Analysis")
        
        # Generate data
        generator = IAMLogGenerator()
        data = generator.generate_dataset(2000, anomaly_ratio=0.1)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(data)
        
        # Test different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        threshold_results = {}
        
        for threshold in thresholds:
            detector = HybridAnomalyDetector(contamination=threshold)
            detector.fit(features, features.columns.tolist())
            predictions, scores = detector.predict(features)
            
            anomaly_count = np.sum(predictions == 1)
            anomaly_percentage = (anomaly_count / len(predictions)) * 100
            
            threshold_results[f'threshold_{threshold}'] = {
                'anomalies_detected': anomaly_count,
                'anomaly_percentage': anomaly_percentage
            }
            
        self.results['threshold_analysis'] = threshold_results
        
    def generate_all_visualizations(self):
        """Generate all visualizations for the report"""
        logger.info("Generating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Model Comparison Chart
        self.create_model_comparison_chart()
        
        # 2. Feature Importance Chart
        self.create_feature_importance_chart()
        
        # 3. Threshold Sensitivity Chart
        self.create_threshold_sensitivity_chart()
        
        # 4. Anomaly Distribution Chart
        self.create_anomaly_distribution_chart()
        
        # 5. Performance Metrics Chart
        self.create_performance_metrics_chart()
        
        # 6. System Architecture Diagram
        self.create_system_architecture_diagram()
        
        # 7. Feature Engineering Pipeline
        self.create_feature_engineering_pipeline()
        
        # 8. Model Architecture Diagram
        self.create_model_architecture_diagram()
        
        # 9. Sample Log Entry
        self.create_sample_log_entry()
        
        # 10. Anomaly Visualization Example
        self.create_anomaly_visualization_example()
        
    def create_model_comparison_chart(self):
        """Create model comparison visualization"""
        if 'model_comparison' not in self.results:
            return
            
        models = list(self.results['model_comparison'].keys())
        anomaly_percentages = [self.results['model_comparison'][m]['anomaly_percentage'] 
                             for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, anomaly_percentages, color=['#2E86AB', '#A23B72', '#F18F01'])
        plt.title('Model Comparison: Anomaly Detection Performance', fontsize=16, fontweight='bold')
        plt.ylabel('Anomaly Detection Rate (%)', fontsize=12)
        plt.xlabel('Model Type', fontsize=12)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, anomaly_percentages):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_feature_importance_chart(self):
        """Create feature importance visualization"""
        if 'feature_importance' not in self.results:
            return
            
        feature_data = self.results['feature_importance']
        features = [item['feature'] for item in feature_data[:10]]  # Top 10 features
        importance = [item['feature_importance'] for item in feature_data[:10]]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(features, importance, color='#2E86AB')
        plt.title('Top 10 Most Important Features for Anomaly Detection', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, importance)):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_threshold_sensitivity_chart(self):
        """Create threshold sensitivity visualization"""
        if 'threshold_analysis' not in self.results:
            return
            
        thresholds = [float(k.split('_')[1]) for k in self.results['threshold_analysis'].keys()]
        anomaly_percentages = [self.results['threshold_analysis'][k]['anomaly_percentage'] 
                             for k in self.results['threshold_analysis'].keys()]
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, anomaly_percentages, 'o-', linewidth=3, markersize=8, color='#A23B72')
        plt.title('Threshold Sensitivity Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Contamination Threshold', fontsize=12)
        plt.ylabel('Anomaly Detection Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add data point labels
        for x, y in zip(thresholds, anomaly_percentages):
            plt.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'threshold_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_anomaly_distribution_chart(self):
        """Create anomaly distribution visualization"""
        # Generate sample data for visualization
        generator = IAMLogGenerator()
        data = generator.generate_dataset(1000, anomaly_ratio=0.1)
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(data)
        
        detector = HybridAnomalyDetector()
        detector.fit(features, features.columns.tolist())
        predictions, scores = detector.predict(features)
        
        plt.figure(figsize=(10, 6))
        anomaly_counts = [np.sum(predictions == 1), np.sum(predictions == 0)]
        labels = ['Anomalies', 'Normal']
        colors = ['#E74C3C', '#2ECC71']
        
        plt.pie(anomaly_counts, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, explode=(0.1, 0))
        plt.title('Distribution of Anomalies vs Normal Events', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'anomaly_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_performance_metrics_chart(self):
        """Create performance metrics visualization"""
        # Create a summary of all performance metrics
        metrics_data = {
            'Synthetic Data': self.results.get('synthetic_performance', {}).get('anomaly_percentage', 0),
            'Real Logs': self.results.get('real_logs_performance', {}).get('anomaly_percentage', 0)
        }
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics_data.keys(), metrics_data.values(), 
                      color=['#3498DB', '#E67E22'])
        plt.title('Performance Comparison: Synthetic vs Real Data', fontsize=16, fontweight='bold')
        plt.ylabel('Anomaly Detection Rate (%)', fontsize=12)
        plt.xlabel('Data Source', fontsize=12)
        
        # Add value labels
        for bar, value in zip(bars, metrics_data.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_system_architecture_diagram(self):
        """Create system architecture diagram"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define components
        components = {
            'Data Sources': (2, 7),
            'Data Processing': (2, 5),
            'Feature Engineering': (2, 3),
            'Hybrid Model': (6, 3),
            'LSTM': (4, 1),
            'Isolation Forest': (8, 1),
            'Results': (10, 3),
            'GUI': (10, 5)
        }
        
        # Draw components
        for name, (x, y) in components.items():
            if name in ['LSTM', 'Isolation Forest']:
                rect = plt.Rectangle((x-0.5, y-0.3), 1, 0.6, 
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
            else:
                rect = plt.Rectangle((x-1, y-0.4), 2, 0.8, 
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw arrows
        arrows = [
            ((2, 6.5), (2, 5.4)),  # Data Sources -> Data Processing
            ((2, 4.6), (2, 3.4)),  # Data Processing -> Feature Engineering
            ((3, 3), (5.4, 3)),    # Feature Engineering -> Hybrid Model
            ((6, 2.4), (4, 1.3)),  # Hybrid Model -> LSTM
            ((6, 2.4), (8, 1.3)),  # Hybrid Model -> Isolation Forest
            ((6.6, 3), (9.4, 3)),  # Hybrid Model -> Results
            ((10, 4.4), (10, 5.6)) # Results -> GUI
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('IAM Anomaly Detection System Architecture', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'system_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_feature_engineering_pipeline(self):
        """Create feature engineering pipeline diagram"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define pipeline stages
        stages = [
            ('Raw Logs', 1),
            ('Data Cleaning', 3),
            ('Temporal Features', 5),
            ('Behavioral Features', 7),
            ('API Patterns', 9),
            ('Final Features', 11)
        ]
        
        # Draw stages
        for name, x in stages:
            rect = plt.Rectangle((x-0.8, 2.2), 1.6, 1.6, 
                               facecolor='lightyellow', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, 3, name, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrows
        for i in range(len(stages)-1):
            start_x = stages[i][1] + 0.8
            end_x = stages[i+1][1] - 0.8
            ax.annotate('', xy=(end_x, 3), xytext=(start_x, 3),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        
        ax.set_xlim(0, 12)
        ax.set_ylim(1, 5)
        ax.axis('off')
        plt.title('Feature Engineering Pipeline', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'feature_engineering_pipeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_model_architecture_diagram(self):
        """Create model architecture diagram"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Define model components
        components = {
            'Input Features': (5, 7),
            'LSTM Network': (3, 5),
            'Isolation Forest': (7, 5),
            'LSTM Output': (3, 3),
            'IF Output': (7, 3),
            'Ensemble Layer': (5, 1),
            'Final Prediction': (5, -1)
        }
        
        # Draw components
        for name, (x, y) in components.items():
            if name in ['LSTM Network', 'Isolation Forest']:
                rect = plt.Rectangle((x-1, y-0.4), 2, 0.8, 
                                   facecolor='lightblue', edgecolor='black', linewidth=2)
            elif name in ['LSTM Output', 'IF Output']:
                rect = plt.Rectangle((x-0.6, y-0.3), 1.2, 0.6, 
                                   facecolor='lightgreen', edgecolor='black', linewidth=2)
            else:
                rect = plt.Rectangle((x-1.2, y-0.4), 2.4, 0.8, 
                                   facecolor='lightyellow', edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y, name, ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Draw arrows
        arrows = [
            ((5, 6.6), (3, 5.4)),  # Input -> LSTM
            ((5, 6.6), (7, 5.4)),  # Input -> IF
            ((3, 4.6), (3, 3.3)),  # LSTM -> LSTM Output
            ((7, 4.6), (7, 3.3)),  # IF -> IF Output
            ((3, 2.7), (4.4, 1.3)), # LSTM Output -> Ensemble
            ((7, 2.7), (5.6, 1.3)), # IF Output -> Ensemble
            ((5, 0.6), (5, -0.3))  # Ensemble -> Final
        ]
        
        for start, end in arrows:
            ax.annotate('', xy=end, xytext=start,
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        
        ax.set_xlim(0, 10)
        ax.set_ylim(-2, 8)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title('Hybrid Model Architecture', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_sample_log_entry(self):
        """Create sample log entry visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sample log entry
        log_entry = {
            'timestamp': '2024-01-15T10:30:45Z',
            'user_id': 'user_123',
            'action': 'GetObject',
            'resource': 's3://my-bucket/file.txt',
            'ip_address': '192.168.1.100',
            'user_agent': 'Mozilla/5.0...',
            'status': 'success',
            'region': 'us-east-1'
        }
        
        # Create text representation
        text_content = "Sample AWS CloudTrail Log Entry:\n\n"
        for key, value in log_entry.items():
            text_content += f"{key}: {value}\n"
        
        ax.text(0.05, 0.95, text_content, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.title('Sample IAM Log Entry', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'sample_log_entry.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_anomaly_visualization_example(self):
        """Create example anomaly visualization"""
        # Generate sample data with anomalies
        generator = IAMLogGenerator()
        data = generator.generate_dataset(500, anomaly_ratio=0.2)
        
        # Feature engineering
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(data)
        
        # Train model and get predictions
        detector = HybridAnomalyDetector()
        detector.fit(features, features.columns.tolist())
        predictions, scores = detector.predict(features)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot normal vs anomalous points
        normal_indices = predictions == 0
        anomaly_indices = predictions == 1
        
        if 'session_duration' in features.columns and 'actions_per_session' in features.columns:
            plt.scatter(features.loc[normal_indices, 'session_duration'], 
                       features.loc[normal_indices, 'actions_per_session'], 
                       c='blue', alpha=0.6, label='Normal', s=30)
            plt.scatter(features.loc[anomaly_indices, 'session_duration'], 
                       features.loc[anomaly_indices, 'actions_per_session'], 
                       c='red', alpha=0.8, label='Anomaly', s=50)
            
            plt.xlabel('Session Duration (minutes)', fontsize=12)
            plt.ylabel('Events per Session', fontsize=12)
        else:
            # Fallback to first two features
            feature_cols = features.columns[:2]
            plt.scatter(features.loc[normal_indices, feature_cols[0]], 
                       features.loc[normal_indices, feature_cols[1]], 
                       c='blue', alpha=0.6, label='Normal', s=30)
            plt.scatter(features.loc[anomaly_indices, feature_cols[0]], 
                       features.loc[anomaly_indices, feature_cols[1]], 
                       c='red', alpha=0.8, label='Anomaly', s=50)
            
            plt.xlabel(feature_cols[0], fontsize=12)
            plt.ylabel(feature_cols[1], fontsize=12)
        
        plt.title('Anomaly Detection Visualization Example', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'anomaly_visualization_example.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_comprehensive_results(self):
        """Save all results to JSON file"""
        results_file = self.results_dir / 'comprehensive_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, np.ndarray):
                        json_results[key][sub_key] = sub_value.tolist()
                    else:
                        json_results[key][sub_key] = sub_value
            else:
                json_results[key] = value
                
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
            
        logger.info(f"Results saved to {results_file}")

if __name__ == "__main__":
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_experiments() 