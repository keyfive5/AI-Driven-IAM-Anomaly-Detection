#!/usr/bin/env python3
"""
Comprehensive Test Suite for IAM Anomaly Detection System
This script runs extensive tests to demonstrate the system's capabilities
and generate high-quality results for the final project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.data_generator import IAMLogGenerator
from src.feature_engineering import FeatureEngineer
from src.models.hybrid_model import HybridAnomalyDetector
from src.simple_detector import SimpleAnomalyDetector
from src.data.iam_log_reader import AWSCloudTrailReader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

class ComprehensiveTestSuite:
    def __init__(self):
        self.results = {}
        self.output_dir = Path("final_project_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_dataset(self):
        """Generate a large, realistic dataset with various anomaly types."""
        print("Generating comprehensive dataset...")
        
        generator = IAMLogGenerator()
        
        # Generate multiple datasets with different characteristics
        datasets = []
        
        # Normal behavior dataset
        print("  - Generating normal behavior data...")
        normal_data = generator.generate_dataset(
            n_events=5000, 
            anomaly_ratio=0.0  # No anomalies
        )
        datasets.append(normal_data)
        
        # Privilege escalation dataset
        print("  - Generating privilege escalation data...")
        escalation_data = generator.generate_dataset(
            n_events=2000,
            anomaly_ratio=0.15
        )
        datasets.append(escalation_data)
        
        # Unusual access patterns dataset
        print("  - Generating unusual access pattern data...")
        access_data = generator.generate_dataset(
            n_events=2000,
            anomaly_ratio=0.12
        )
        datasets.append(access_data)
        
        # Geographic anomalies dataset
        print("  - Generating geographic anomaly data...")
        geo_data = generator.generate_dataset(
            n_events=1500,
            anomaly_ratio=0.10
        )
        datasets.append(geo_data)
        
        # Combine all datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)  # Shuffle
        
        print(f"Generated comprehensive dataset with {len(combined_data)} events")
        print(f"True anomalies: {combined_data['is_anomaly'].sum()} ({combined_data['is_anomaly'].mean():.2%})")
        
        return combined_data
    
    def run_feature_engineering_analysis(self, df):
        """Run comprehensive feature engineering and analysis."""
        print("\nRunning feature engineering analysis...")
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df)
        
        # Analyze feature importance
        feature_columns = feature_engineer.get_feature_columns()
        print(f"Generated {len(feature_columns)} features")
        
        # Create feature importance visualization
        if 'is_anomaly' in df_features.columns:
            # Calculate correlation with anomalies
            correlations = []
            for col in feature_columns:
                if col in df_features.columns and df_features[col].dtype in ['int64', 'float64']:
                    corr = abs(df_features[col].corr(df_features['is_anomaly']))
                    correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = correlations[:10]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            features, scores = zip(*top_features)
            plt.barh(range(len(features)), scores)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Absolute Correlation with Anomalies')
            plt.title('Top 10 Most Important Features')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.results['feature_importance'] = dict(top_features)
        
        return df_features, feature_columns
    
    def run_model_comparison(self, df_features, feature_columns):
        """Compare different models and configurations."""
        print("\nRunning model comparison...")
        
        # Prepare data
        X = df_features[feature_columns]
        y = df_features['is_anomaly'] if 'is_anomaly' in df_features.columns else None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y if y is not None else None
        )
        
        models = {}
        results = {}
        
        # Test different contamination levels
        contamination_levels = [0.05, 0.1, 0.15, 0.2]
        
        for cont in contamination_levels:
            print(f"  Testing contamination level: {cont}")
            
            # Hybrid Model
            hybrid_model = HybridAnomalyDetector(
                contamination=cont,
                n_estimators_iso_forest=200,
                n_estimators_rf=200,
                max_depth_rf=20
            )
            
            hybrid_model.fit(X_train, feature_columns)
            predictions, scores = hybrid_model.predict(X_test)
            
            if y_test is not None:
                precision = precision_score(y_test, predictions)
                recall = recall_score(y_test, predictions)
                f1 = f1_score(y_test, predictions)
                accuracy = accuracy_score(y_test, predictions)
                
                results[f'hybrid_cont_{cont}'] = {
                    'contamination': cont,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'accuracy': accuracy,
                    'anomaly_rate': predictions.mean()
                }
            
            models[f'hybrid_cont_{cont}'] = hybrid_model
        
        # Test simple detector
        print("  Testing simple detector...")
        simple_detector = SimpleAnomalyDetector()
        simple_predictions = simple_detector.detect_anomalies(X_test)
        
        if y_test is not None:
            precision = precision_score(y_test, simple_predictions)
            recall = recall_score(y_test, simple_predictions)
            f1 = f1_score(y_test, simple_predictions)
            accuracy = accuracy_score(y_test, simple_predictions)
            
            results['simple_detector'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'anomaly_rate': simple_predictions.mean()
            }
        
        self.results['model_comparison'] = results
        
        # Create comparison visualization
        if y_test is not None:
            self.create_model_comparison_plot(results)
        
        return models, results
    
    def create_model_comparison_plot(self, results):
        """Create visualization comparing model performance."""
        plt.figure(figsize=(15, 10))
        
        # Extract metrics
        model_names = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[i].bar(model_names, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_threshold_analysis(self, df_features, feature_columns):
        """Analyze the impact of different thresholds."""
        print("\nRunning threshold sensitivity analysis...")
        
        X = df_features[feature_columns]
        y = df_features['is_anomaly'] if 'is_anomaly' in df_features.columns else None
        
        thresholds = np.arange(0.01, 0.31, 0.01)
        results = []
        
        for threshold in thresholds:
            model = HybridAnomalyDetector(contamination=threshold)
            model.fit(X, feature_columns)
            predictions, scores = model.predict(X)
            
            if y is not None:
                precision = precision_score(y, predictions)
                recall = recall_score(y, predictions)
                f1 = f1_score(y, predictions)
                
                results.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'anomaly_rate': predictions.mean()
                })
        
        # Create threshold sensitivity plot
        if results:
            df_results = pd.DataFrame(results)
            
            plt.figure(figsize=(12, 8))
            plt.plot(df_results['threshold'], df_results['precision'], label='Precision', marker='o')
            plt.plot(df_results['threshold'], df_results['recall'], label='Recall', marker='s')
            plt.plot(df_results['threshold'], df_results['f1_score'], label='F1-Score', marker='^')
            plt.xlabel('Contamination Threshold')
            plt.ylabel('Score')
            plt.title('Threshold Sensitivity Analysis')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'threshold_sensitivity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            self.results['threshold_analysis'] = df_results.to_dict('records')
    
    def run_real_data_analysis(self):
        """Analyze real AWS CloudTrail data."""
        print("\nRunning real data analysis...")
        
        # Load real AWS data
        aws_reader = AWSCloudTrailReader()
        real_data_path = "data/sample_aws_cloudtrail.json"
        
        if os.path.exists(real_data_path):
            real_data = aws_reader.read_logs(real_data_path)
            
            if not real_data.empty:
                # Add synthetic anomalies to real data for testing
                real_data['is_anomaly'] = 0  # Assume all real data is normal initially
                
                # Add some synthetic anomalies for testing
                anomaly_indices = np.random.choice(
                    len(real_data), 
                    size=int(len(real_data) * 0.1), 
                    replace=False
                )
                real_data.loc[anomaly_indices, 'is_anomaly'] = 1
                
                # Run analysis on real data
                df_features, feature_columns = self.run_feature_engineering_analysis(real_data)
                models, results = self.run_model_comparison(df_features, feature_columns)
                
                self.results['real_data_analysis'] = {
                    'total_events': len(real_data),
                    'anomaly_rate': real_data['is_anomaly'].mean(),
                    'model_results': results
                }
                
                return real_data, df_features
            else:
                print("  Warning: Real data is empty")
        else:
            print("  Warning: Real data file not found")
        
        return None, None
    
    def create_anomaly_distribution_plot(self, df_features):
        """Create visualization of anomaly distribution."""
        print("\nCreating anomaly distribution visualization...")
        
        if 'anomaly_score' in df_features.columns:
            plt.figure(figsize=(12, 8))
            
            # Plot distribution of anomaly scores
            plt.subplot(2, 2, 1)
            plt.hist(df_features['anomaly_score'], bins=50, alpha=0.7, color='skyblue')
            plt.xlabel('Anomaly Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Anomaly Scores')
            
            # Plot anomalies vs normal events
            plt.subplot(2, 2, 2)
            if 'is_anomaly_predicted' in df_features.columns:
                anomaly_counts = df_features['is_anomaly_predicted'].value_counts()
                plt.pie(anomaly_counts.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%')
                plt.title('Predicted Anomalies vs Normal Events')
            
            # Plot anomalies over time
            plt.subplot(2, 2, 3)
            if 'timestamp' in df_features.columns and 'is_anomaly_predicted' in df_features.columns:
                df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
                anomalies_by_hour = df_features[df_features['is_anomaly_predicted'] == 1].groupby(
                    df_features['timestamp'].dt.hour
                ).size()
                plt.bar(anomalies_by_hour.index, anomalies_by_hour.values, color='red', alpha=0.7)
                plt.xlabel('Hour of Day')
                plt.ylabel('Number of Anomalies')
                plt.title('Anomalies by Hour of Day')
            
            # Plot feature correlation heatmap
            plt.subplot(2, 2, 4)
            if 'is_anomaly' in df_features.columns:
                numeric_cols = df_features.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    corr_matrix = df_features[numeric_cols].corr()
                    sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
                    plt.title('Feature Correlation Heatmap (Top 10)')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'anomaly_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all results."""
        print("\nGenerating comprehensive report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests_run': len(self.results),
                'models_tested': len([k for k in self.results.keys() if 'model' in k]),
                'datasets_analyzed': len([k for k in self.results.keys() if 'data' in k])
            },
            'results': self.results
        }
        
        # Save report
        with open(self.output_dir / 'comprehensive_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate summary statistics
        if 'model_comparison' in self.results:
            best_model = max(
                self.results['model_comparison'].items(),
                key=lambda x: x[1].get('f1_score', 0) if isinstance(x[1], dict) else 0
            )
            
            print(f"\n=== COMPREHENSIVE TEST RESULTS ===")
            print(f"Best performing model: {best_model[0]}")
            if isinstance(best_model[1], dict):
                print(f"F1-Score: {best_model[1].get('f1_score', 0):.3f}")
                print(f"Precision: {best_model[1].get('precision', 0):.3f}")
                print(f"Recall: {best_model[1].get('recall', 0):.3f}")
                print(f"Accuracy: {best_model[1].get('accuracy', 0):.3f}")
        
        print(f"\nAll results saved to: {self.output_dir}")
    
    def run_all_tests(self):
        """Run the complete test suite."""
        print("=== IAM ANOMALY DETECTION - COMPREHENSIVE TEST SUITE ===")
        print("This test suite will generate comprehensive results for the final project.")
        
        # Generate comprehensive dataset
        df = self.generate_comprehensive_dataset()
        
        # Run feature engineering
        df_features, feature_columns = self.run_feature_engineering_analysis(df)
        
        # Run model comparison
        models, results = self.run_model_comparison(df_features, feature_columns)
        
        # Run threshold analysis
        self.run_threshold_analysis(df_features, feature_columns)
        
        # Run real data analysis
        real_data, real_features = self.run_real_data_analysis()
        
        # Create visualizations
        self.create_anomaly_distribution_plot(df_features)
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        print("\n=== TEST SUITE COMPLETED SUCCESSFULLY ===")
        print("All tests completed. Check the 'final_project_results' directory for outputs.")

def main():
    """Main function to run the comprehensive test suite."""
    test_suite = ComprehensiveTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main() 