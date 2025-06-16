# IAM Anomaly Detection

A comprehensive IAM (Identity and Access Management) anomaly detection system that analyzes logs from various sources to identify suspicious activities and potential security threats.

## Features

- Support for multiple log sources:
  - AWS CloudTrail
  - Azure Activity Logs
  - CyberArk Logs
  - Synthetic data generation for testing
- Advanced feature engineering
- Hybrid anomaly detection using multiple algorithms
- Comprehensive logging system
- User-friendly GUI interface
- Real-time analysis and visualization

## Logging System

The application implements a comprehensive logging system that captures important events, errors, and debugging information. The logging system includes:

### Log Levels

- DEBUG: Detailed information for debugging
- INFO: General information about program execution
- WARNING: Indicate a potential problem
- ERROR: A more serious problem
- CRITICAL: A critical problem that may prevent the program from running

### Log Components

- Data Processing: Logs related to data loading, parsing, and preprocessing
- Model: Logs related to model training, prediction, and evaluation
- GUI: Logs related to user interface interactions
- Analysis: Logs related to feature engineering and analysis

### Log Storage

- Logs are stored in the `logs` directory
- Daily log files with format: `iam_anomaly_detection_YYYYMMDD.log`
- Log rotation: Files are rotated when they reach 10MB
- Maximum of 5 backup files are kept

### Log Format

Each log entry includes:
- Timestamp
- Component name
- Log level
- File name and line number
- Message

Example:
```
2024-01-21 10:00:00,123 - data - INFO - [iam_log_reader.py:45] - Reading logs from aws_cloudtrail.json
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/iam-anomaly-detection.git
cd iam-anomaly-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Select your data source:
   - AWS CloudTrail logs
   - Azure Activity logs
   - CyberArk logs
   - Synthetic data

3. Configure analysis parameters:
   - Number of events
   - Anomaly detection threshold
   - Feature selection

4. Run the analysis and view results in the GUI

## Logging Configuration

The logging system can be configured in `src/utils/logging_config.py`. Key settings include:

- Log directory location
- Log file size limits
- Number of backup files
- Log format
- Log levels for different components

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 