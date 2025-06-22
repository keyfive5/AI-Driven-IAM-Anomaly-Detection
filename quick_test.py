#!/usr/bin/env python3
"""
Quick Test Script for IAM Anomaly Detection System
Generates manageable datasets and produces good statistics for final project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

class QuickTestSuite:
    def __init__(self):
        self.results = {}
        self.output_dir = Path("final_project_results")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_manageable_dataset(self):
        """Generate a smaller, manageable dataset for testing."""
        print("Generating manageable dataset...")
        
        generator = IAMLogGenerator()
        
        # Generate smaller datasets
        datasets = []
        
        # Normal behavior dataset (smaller)
        print("  - Generating normal behavior data...")
        normal_data = generator.generate_dataset(
            n_events=1000, 
            anomaly_ratio=0.0
        )
        datasets.append(normal_data)
        
        # Anomalous dataset
        print("  - Generating anomalous data...")
        anomalous_data = generator.generate_dataset(
            n_events=500,
            anomaly_ratio=0.2
        )
        datasets.append(anomalous_data)
        
        # Combine datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        print(f"Generated dataset with {len(combined_data)} events")
        print(f"True anomalies: {combined_data['is_anomaly'].sum()} ({combined_data['is_anomaly'].mean():.2%})")
        
        return combined_data
    
    def run_feature_engineering(self, df):
        """Run feature engineering on the dataset."""
        print("\nRunning feature engineering...")
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df)
        
        feature_columns = feature_engineer.get_feature_columns()
        print(f"Generated {len(feature_columns)} features")
        
        return df_features, feature_columns
    
    def run_model_evaluation(self, df_features, feature_columns):
        """Evaluate different models and configurations."""
        print("\nRunning model evaluation...")
        
        # Prepare data
        X = df_features[feature_columns]
        y = df_features['is_anomaly']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        results = {}
        
        # Test different contamination levels
        contamination_levels = [0.05, 0.1, 0.15, 0.2]
        
        for cont in contamination_levels:
            print(f"  Testing contamination level: {cont}")
            
            # Hybrid Model
            hybrid_model = HybridAnomalyDetector(
                contamination=cont,
                n_estimators_iso_forest=100,
                n_estimators_rf=100,
                max_depth_rf=15
            )
            
            hybrid_model.fit(X_train, feature_columns)
            predictions, scores = hybrid_model.predict(X_test)
            
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
        
        # Test simple detector
        print("  Testing simple detector...")
        simple_detector = SimpleAnomalyDetector()
        simple_predictions = simple_detector.detect_anomalies(X_test)
        
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
        
        # Create visualizations
        self.create_performance_visualizations(results)
        
        return results
    
    def create_performance_visualizations(self, results):
        """Create performance comparison visualizations."""
        print("\nCreating performance visualizations...")
        
        # Model comparison plot
        plt.figure(figsize=(15, 10))
        
        model_names = list(results.keys())
        metrics = ['precision', 'recall', 'f1_score', 'accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in model_names]
            axes[i].bar(model_names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance analysis
        self.analyze_feature_importance()
    
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance."""
        print("\nAnalyzing feature importance...")
        
        # Generate a small dataset for feature importance
        generator = IAMLogGenerator()
        df = generator.generate_dataset(n_events=500, anomaly_ratio=0.15)
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.engineer_features(df)
        feature_columns = feature_engineer.get_feature_columns()
        
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
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
        plt.barh(range(len(features)), scores, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Absolute Correlation with Anomalies')
        plt.title('Top 10 Most Important Features for Anomaly Detection')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.results['feature_importance'] = dict(top_features)
    
    def generate_final_report(self):
        """Generate a comprehensive report of results."""
        print("\nGenerating final report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests_run': len(self.results),
                'models_tested': len(self.results.get('model_comparison', {})),
                'datasets_analyzed': 1
            },
            'results': self.results
        }
        
        # Save report
        with open(self.output_dir / 'quick_test_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        if 'model_comparison' in self.results:
            best_model = max(
                self.results['model_comparison'].items(),
                key=lambda x: x[1].get('f1_score', 0)
            )
            
            print(f"\n=== QUICK TEST RESULTS ===")
            print(f"Best performing model: {best_model[0]}")
            print(f"F1-Score: {best_model[1].get('f1_score', 0):.3f}")
            print(f"Precision: {best_model[1].get('precision', 0):.3f}")
            print(f"Recall: {best_model[1].get('recall', 0):.3f}")
            print(f"Accuracy: {best_model[1].get('accuracy', 0):.3f}")
            print(f"Anomaly Rate: {best_model[1].get('anomaly_rate', 0):.3f}")
        
        print(f"\nAll results saved to: {self.output_dir}")
    
    def run_quick_test(self):
        """Run the complete quick test suite."""
        print("=== IAM ANOMALY DETECTION - QUICK TEST SUITE ===")
        print("This test suite will generate good statistics for the final project.")
        
        # Generate dataset
        df = self.generate_manageable_dataset()
        
        # Run feature engineering
        df_features, feature_columns = self.run_feature_engineering(df)
        
        # Run model evaluation
        results = self.run_model_evaluation(df_features, feature_columns)
        
        # Generate report
        self.generate_final_report()
        
        print("\n=== QUICK TEST SUITE COMPLETED SUCCESSFULLY ===")
        print("Good statistics generated for final project demonstration!")

def main():
    """Main function to run the quick test suite."""
    test_suite = QuickTestSuite()
    test_suite.run_quick_test()

if __name__ == "__main__":
    main() 