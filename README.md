# AI-Driven Identity Access Management System

This project implements an advanced AI-driven Identity and Access Management (IAM) system that combines supervised and unsupervised learning approaches to detect anomalous access patterns and potential security threats.

## Introduction

Modern cloud and enterprise environments generate vast amounts of IAM logs, crucial for tracking user access and maintaining security. Manual analysis of these logs is infeasible, making AI-driven techniques essential for detecting automated attacks, compromised accounts, and insider threats. This project addresses this challenge by developing an AI-based system designed to classify access requests as legitimate or fraudulent and identify anomalous access behaviors.

## Features

-   **Synthetic IAM Log Generation**: Generates realistic IAM log data, including various types of anomalies (e.g., access outside working hours, unauthorized actions, unusual IP addresses).
-   **Advanced Feature Engineering**: Extracts rich, domain-specific features from raw logs, including time-based, IP-based, behavioral, and session-based attributes.
-   **Hybrid Anomaly Detection Model**: Employs a robust combination of:
    -   **Isolation Forest**: For efficient unsupervised outlier detection, identifying events that deviate significantly from normal patterns.
    -   **Random Forest Classifier**: For supervised classification, trained to distinguish between legitimate and fraudulent access requests.
-   **Interactive GUI**: A user-friendly Tkinter-based graphical interface for:
    -   Adjusting data generation parameters (number of events, users, roles, actions).
    -   Running the anomaly detection analysis.
    -   Viewing real-time status updates and model performance metrics.
    -   Displaying dynamic visualizations of anomaly scores and patterns.
-   **Comprehensive Evaluation**: Calculates key performance metrics (Precision, Recall, F1-score) to assess detection quality.
-   **Results Export**: Automatically saves detailed anomaly detection results and visualizations for further analysis.

## Project Structure

```
├── requirements.txt
├── README.md
├── src/
│   ├── data_generator.py          # Generates synthetic IAM log data
│   ├── feature_engineering.py     # Extracts and processes features from logs
│   ├── models/
│   │   └── hybrid_model.py        # Implements the Isolation Forest + Random Forest hybrid model
│   └── main.py                    # Main script with GUI to run the full system
└── output/                        # Directory to store generated results and plots
    ├── anomaly_analysis.png       # Visualization of anomaly patterns
    └── anomaly_results.csv        # Detailed CSV of detected anomalies
```

## Setup

1.  **Clone the repository** (once available on GitHub).
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment**:
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Demo

Once setup is complete:

1.  **Open your terminal or command prompt.**
2.  **Navigate to the project's root directory** (`cursor 6670 project`).
3.  **Run the main script**:
    ```bash
    python src/main.py
    ```
4.  **Interact with the GUI**:
    *   A window titled "IAM Anomaly Detection" will appear.
    *   On the left "Controls" panel, you can adjust parameters like "Number of Events", "Number of Users", "Number of Roles", and "Number of Actions" to generate different sizes and complexities of synthetic data.
    *   Click the **"Run Analysis"** button to start the process.
    *   The "Status" text area will display real-time updates on data generation, feature extraction, model training, and prediction.
    *   The "Visualization" panel on the right will update with four interactive plots once the analysis is complete.

## Understanding the Output & Visualizations

After clicking "Run Analysis", the GUI will provide the following:

### Status Panel Messages:
-   **Progress updates**: Messages like "Generating IAM logs...", "Extracting features...", "Training hybrid anomaly detection model...", "Making predictions..." indicate the current step.
-   **Model Performance**: Precision, Recall, and F1 Score for the anomaly detection. These metrics assess how well the model identifies true anomalies while minimizing false alarms.
-   **Results Saved Confirmation**: Indicates that `anomaly_results.csv` and `anomaly_analysis.png` have been saved in the `output/` directory.

### Visualization Panel (`anomaly_analysis.png`):
The GUI displays four key plots:
1.  **Distribution of Anomaly Scores**: A histogram showing the distribution of anomaly scores assigned by the model. Higher scores generally indicate a higher likelihood of an anomaly.
2.  **Anomalies by Hour of Day**: A bar chart showing the frequency of detected anomalies across different hours, highlighting potential time-based attack patterns.
3.  **Top 10 Users with Anomalies**: Identifies users who generated the most anomalous activities, helping prioritize investigations.
4.  **Anomaly Scores Over Time**: A scatter plot showing anomaly scores against timestamps, with anomalous points typically highlighted in a different color. This helps visualize trends and clusters of suspicious activity.

## Current Model Performance & Limitations

You may observe low Precision, Recall, and F1 Scores (e.g., around 0.060). This is expected at this stage due to several factors, reflecting a foundational prototype:

-   **Simplified Synthetic Data**: The current `data_generator.py` creates relatively straightforward anomalies. Real-world IAM logs exhibit far more complex and subtle anomalous patterns.
-   **Baseline Model Implementation**: While Isolation Forest and Random Forest are powerful, their current application is a basic setup without extensive tuning or sophisticated layering.
-   **Lack of Hyperparameter Tuning**: The models are running with default parameters. Optimal performance requires fine-tuning hyperparameters specific to the dataset.
-   **Class Imbalance Handling**: Anomalies are inherently rare events in real logs. Without explicit techniques to address this class imbalance (e.g., oversampling the minority class, using class weights), models often struggle to effectively learn and detect anomalies.

These limitations are acknowledged and serve as clear targets for future improvements.

## Future Work & Enhancements

Our project proposal outlines ambitious steps to significantly enhance the system's capabilities and performance, directly addressing the feedback received and increasing its creativity/novelty:

1.  **Extensive Feature Engineering**:
    *   **Advanced Temporal Features**: Incorporate rolling window statistics (e.g., average logins per user in the last hour/day), sequence patterns (e.g., common sequences of actions), and more sophisticated time-of-day encodings.
    *   **Behavioral Baselines**: Develop dynamic "normal" behavioral profiles for each user/role and calculate real-time deviations as a feature.
    *   **Contextual Feature Expansion**: Integrate concepts like geo-location changes, concurrent logins from disparate locations, and unusual resource access.

2.  **Hybrid and More Complex Algorithms**:
    *   **Reintroduce Deep Learning (LSTM Autoencoders)**: Implement LSTM Autoencoders to learn normal sequences of user actions. Large reconstruction errors will indicate anomalous sequences, effectively capturing temporal patterns that traditional tree-based models might miss.
    *   **Sophisticated Ensemble Methods**: Explore stacking or blending different anomaly detection algorithms (e.g., Isolation Forest, One-Class SVM, LOF, LSTM Autoencoder) to leverage their strengths and improve overall detection accuracy and robustness.
    *   **Adaptive Thresholding**: Implement dynamic thresholding for anomaly scores based on real-time data characteristics or feedback loops.

3.  **Hyperparameter Optimization**: Conduct rigorous hyperparameter tuning using techniques like Grid Search or Randomized Search to find the optimal configuration for all models, maximizing precision while maintaining acceptable recall.

4.  **Robust Class Imbalance Handling**: Implement advanced strategies such as SMOTE, ADASYN, or weighted loss functions during model training to ensure the model effectively learns from and detects the rare anomalous events.

5.  **Interactive Visualizations & Reporting**: Further enhance the GUI with more interactive elements, detailed drill-down capabilities for anomalies, and real-time dashboarding.

By implementing these planned enhancements, we aim to demonstrate a highly effective and adaptive AI-driven IAM system capable of significantly enhancing security.

## Team Members

-   Muhammad Zafar (1125402) - Team Lead, Data Engineering, Modeling and Evaluation
-   Ronin Furtado (1357823) - Feature Engineering, Literature Review, Visualization
-   Safiulla Syed Arshad (1354086) - Data Collection Support, Tool Configuration, Presentation Prep 