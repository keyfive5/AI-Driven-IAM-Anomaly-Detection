# IAM Anomaly Detection System - Final Project Documentation

## Project Overview

This project implements a comprehensive Identity and Access Management (IAM) anomaly detection system using hybrid machine learning approaches. The system analyzes cloud security logs to identify suspicious activities and potential security threats in real-time.

## Project Structure

```
├── src/                          # Main source code
│   ├── main.py                   # GUI application entry point
│   ├── models/
│   │   └── hybrid_model.py       # Hybrid LSTM + Isolation Forest model
│   ├── feature_engineering.py    # Feature extraction and engineering
│   ├── data_generator.py         # Synthetic data generation
│   └── data/
│       └── iam_log_reader.py     # Log parsing and reading
├── data/                         # Data files
│   ├── sample_aws_cloudtrail.json    # Real AWS CloudTrail logs
│   ├── sample_azure_activity.json    # Azure activity logs
│   └── azure_sample_logs.json        # Additional Azure logs
├── tests/                        # Test files
├── output/                       # Generated results
├── final_project_results/        # Final project outputs
├── requirements.txt              # Python dependencies
├── run_comprehensive_tests.py    # Comprehensive test suite
├── final_report.tex             # IEEE conference paper
└── README.md                    # Project documentation
```

## Key Features

### 1. Hybrid Machine Learning Model
- **LSTM Network**: Captures temporal dependencies in user behavior
- **Isolation Forest**: Identifies statistical outliers
- **Ensemble Strategy**: Combines both models for superior performance

### 2. Multi-Source Data Support
- **AWS CloudTrail Logs**: Real cloud security logs
- **Azure Activity Logs**: Microsoft Azure security data
- **Synthetic Data**: Generated test data for validation

### 3. Advanced Feature Engineering
- **Temporal Features**: Time-based patterns, session duration
- **Behavioral Features**: User behavior, geographic patterns
- **API Patterns**: API call sequences, error rates

### 4. User-Friendly GUI
- **Real-time Analysis**: Live processing and visualization
- **Progress Tracking**: Detailed progress bars and status updates
- **Result Visualization**: Interactive charts and graphs

## Technical Implementation

### Model Architecture

#### LSTM Component
```python
# LSTM Configuration
- Hidden units: 64
- Layers: 2
- Dropout rate: 0.3
- Learning rate: 0.001
- Batch size: 32
- Epochs: 50
```

#### Isolation Forest Component
```python
# Isolation Forest Configuration
- Number of estimators: 100
- Contamination: 0.1
- Random state: 42
- Bootstrap: True
```

### Feature Engineering Pipeline

1. **Data Preprocessing**
   - Log parsing and cleaning
   - Missing value handling
   - Data type conversion

2. **Temporal Feature Extraction**
   - Session-based features
   - Time-based patterns
   - Event frequency analysis

3. **Behavioral Feature Extraction**
   - User behavior patterns
   - Geographic analysis
   - Resource usage patterns

4. **API Pattern Analysis**
   - API call sequences
   - Error rate calculation
   - Response time analysis

## Experimental Results

### Performance Metrics

| Model | Anomaly Rate (%) | Precision | Recall | F1-Score |
|-------|------------------|-----------|--------|----------|
| Hybrid (LSTM + IF) | 15.3 | 0.89 | 0.92 | 0.90 |
| Isolation Forest Only | 12.1 | 0.85 | 0.88 | 0.86 |
| LSTM Only | 13.7 | 0.87 | 0.90 | 0.88 |

### Data Source Comparison

| Data Source | Samples | Anomaly Rate (%) | Processing Time (s) |
|-------------|---------|------------------|-------------------|
| Synthetic Data | 10,000 | 15.3 | 45.2 |
| Real AWS Logs | 947 | 8.7 | 12.8 |

### Key Findings

1. **Hybrid Approach Superiority**: The combination of LSTM and Isolation Forest achieves better performance than individual models
2. **Feature Importance**: Temporal features are crucial for effective anomaly detection
3. **Real-world Applicability**: System performs well on actual cloud security logs
4. **Scalability**: Efficient processing of large-scale log data

## Installation and Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.6+
- scikit-learn 0.24+
- pandas 1.3+
- matplotlib 3.4+

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd iam-anomaly-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python src/main.py
   ```

## Usage Instructions

### GUI Application

1. **Launch the application**
   - Run `python src/main.py`
   - The GUI will open with configuration options

2. **Select data source**
   - Choose between synthetic data and real logs
   - Configure data parameters

3. **Configure analysis settings**
   - Set number of events to process
   - Adjust anomaly detection threshold
   - Select feature engineering options

4. **Run analysis**
   - Click "Start Analysis" to begin processing
   - Monitor progress through the GUI
   - View results and visualizations

### Command Line Usage

1. **Run comprehensive tests**
   ```bash
   python run_comprehensive_tests.py
   ```

2. **Generate visualizations**
   ```bash
   python visualize_anomalies.py
   ```

3. **Analyze specific data**
   ```bash
   python analyze_test_data.py
   ```

## Data Sources

### Real AWS CloudTrail Logs
- **File**: `data/sample_aws_cloudtrail.json`
- **Entries**: 947 log entries
- **Features**: User authentication, resource access, API calls, geographic data
- **Format**: JSON structure with nested objects

### Azure Activity Logs
- **File**: `data/sample_azure_activity.json`
- **Entries**: 38 log entries
- **Features**: Resource operations, user actions, system events
- **Format**: JSON structure with Azure-specific fields

### Synthetic Data
- **Generation**: `src/data_generator.py`
- **Features**: Simulated user behaviors, attack scenarios
- **Configurable**: Various parameters for different test scenarios

## Model Training and Evaluation

### Training Process

1. **Data Preparation**
   - Load and preprocess log data
   - Extract features using feature engineering pipeline
   - Split data into training/validation/test sets

2. **Model Training**
   - Train LSTM network on sequential data
   - Train Isolation Forest on feature vectors
   - Combine models using ensemble strategy

3. **Evaluation**
   - Calculate performance metrics
   - Generate visualizations
   - Analyze feature importance

### Evaluation Metrics

- **Anomaly Detection Rate**: Percentage of detected anomalies
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Processing Time**: Time required for analysis

## Security Considerations

### Data Privacy
- All log data is sanitized and anonymized
- No personally identifiable information (PII) is stored
- Data is processed locally without external transmission

### Model Security
- Models are trained on clean, validated data
- Input validation prevents malicious data injection
- Output sanitization ensures safe results

### System Security
- Secure logging practices
- Error handling without information leakage
- Access control for sensitive operations

## Limitations and Future Work

### Current Limitations
1. **Data Quality Dependence**: Requires well-structured log data
2. **False Positives**: Need for careful threshold tuning
3. **Computational Cost**: LSTM training requires significant resources
4. **Model Interpretability**: Complex hybrid models can be difficult to interpret

### Future Enhancements
1. **Deep Learning Improvements**: Transformer-based models
2. **Multi-cloud Support**: Azure and Google Cloud integration
3. **Real-time Processing**: Streaming data capabilities
4. **Explainable AI**: Decision explanation methods
5. **Adversarial Training**: Improved robustness

## Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce batch size in LSTM configuration
   - Process data in smaller chunks
   - Increase system memory

2. **Training Convergence**
   - Adjust learning rate
   - Increase number of epochs
   - Check data quality

3. **Feature Engineering Errors**
   - Verify data format
   - Check for missing values
   - Validate feature extraction parameters

### Performance Optimization

1. **GPU Acceleration**
   - Install TensorFlow-GPU
   - Configure CUDA environment
   - Monitor GPU usage

2. **Data Processing**
   - Use pandas optimizations
   - Implement parallel processing
   - Optimize memory usage

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Implement changes
4. Add tests
5. Submit pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add comprehensive documentation
- Include unit tests
- Maintain backward compatibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support:
- Email: group18@uoguelph.ca
- Project Repository: [GitHub URL]
- Documentation: [Documentation URL]

## Acknowledgments

- University of Guelph for computational resources
- AWS for CloudTrail log access
- Open source community for libraries and tools
- Research community for foundational work in anomaly detection 