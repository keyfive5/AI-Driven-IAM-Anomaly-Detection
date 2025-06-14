import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Callable, Optional
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, RepeatVector
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.callbacks import Callback

class KerasProgressCallback(Callback):
    def __init__(self, overall_progress_start: int, overall_progress_end: int, total_epochs: int, update_gui_progress_callback: Callable[[int, str], None]):
        super().__init__()
        self.overall_progress_start = overall_progress_start
        self.overall_progress_end = overall_progress_end
        self.total_epochs = total_epochs
        self.update_gui_progress_callback = update_gui_progress_callback

    def on_epoch_end(self, epoch, logs=None):
        lstm_current_progress = (epoch + 1) / self.total_epochs
        
        overall_progress_range = self.overall_progress_end - self.overall_progress_start
        current_overall_progress = self.overall_progress_start + (lstm_current_progress * overall_progress_range)
        
        self.update_gui_progress_callback(int(current_overall_progress), f"Training LSTM (Epoch {epoch+1}/{self.total_epochs})")

class HybridAnomalyDetector:
    def __init__(self, sequence_length: int = 10, lstm_units: int = 64, contamination: float = 0.1):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42
        )
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced' # Add balanced class weight for imbalance
        )
        self.lstm_autoencoder = None
        self.feature_columns = None
        
    def _build_lstm_autoencoder(self, input_shape: Tuple[int, int]) -> Model:
        """Build LSTM Autoencoder model."""
        inputs = Input(shape=input_shape)
        # Encoder
        encoded = LSTM(self.lstm_units, activation='relu', return_sequences=True)(inputs)
        encoded = Dropout(0.2)(encoded)
        encoded = LSTM(self.lstm_units // 2, activation='relu', return_sequences=False)(encoded)
        
        # Decoder
        decoded = RepeatVector(input_shape[0])(encoded)
        decoded = LSTM(self.lstm_units // 2, activation='relu', return_sequences=True)(decoded)
        decoded = Dropout(0.2)(decoded)
        decoded = LSTM(self.lstm_units, activation='relu', return_sequences=True)(decoded)
        decoded = Dense(input_shape[1])(decoded)
        
        autoencoder = Model(inputs, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return autoencoder
    
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """Create sequences for LSTM input."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)
    
    def fit(self, X: pd.DataFrame, feature_columns: List[str], progress_callback: Optional[Callable[[int, str], None]] = None) -> None:
        """Train the hybrid model."""
        self.feature_columns = feature_columns
        X_scaled = self.scaler.fit_transform(X[feature_columns])

        # Train Isolation Forest
        if progress_callback: progress_callback(45, "Fitting Isolation Forest model...")
        self.isolation_forest.fit(X_scaled)
        if progress_callback: progress_callback(50, "Isolation Forest trained.")
        
        # Prepare sequences for LSTM and train LSTM Autoencoder
        # We train LSTM on the *normal* data identified by Isolation Forest to learn normal patterns
        if_scores_train = -self.isolation_forest.score_samples(X_scaled)
        normal_indices = if_scores_train < np.percentile(if_scores_train, (1 - self.contamination) * 100) # Use contamination
        X_normal_sequences = self._create_sequences(X_scaled[normal_indices])

        if len(X_normal_sequences) == 0:
            print("Warning: No normal sequences found for LSTM training. Skipping LSTM training.")
            self.lstm_autoencoder = None # Ensure it's None if not trained
            if progress_callback: progress_callback(80, "LSTM training skipped.") # Adjust end percentage
        else:
            input_shape = (self.sequence_length, len(feature_columns))
            self.lstm_autoencoder = self._build_lstm_autoencoder(input_shape)
            
            # Setup Keras callback for fine-grained progress updates
            total_epochs_lstm = 50 # Match epochs in .fit()
            keras_callback = KerasProgressCallback(50, 80, total_epochs_lstm, progress_callback) # Changed progress range

            if progress_callback: progress_callback(51, "Starting LSTM Autoencoder training...") # Adjusted start after IF
            self.lstm_autoencoder.fit(
                X_normal_sequences, X_normal_sequences,
                epochs=total_epochs_lstm,
                batch_size=32,
                validation_split=0.2,
                verbose=0,
                callbacks=[keras_callback] # Pass the custom callback here
            )
            if progress_callback: progress_callback(80, "LSTM Autoencoder trained.") # Changed end percentage

        # Train Random Forest. Labels for RF are derived from Isolation Forest initial predictions
        if progress_callback: progress_callback(81, "Fitting Random Forest...") # Changed from 69
        threshold_for_rf_labels = np.percentile(if_scores_train, (1 - self.contamination) * 100) # Use contamination
        rf_labels = (if_scores_train > threshold_for_rf_labels).astype(int)
        self.random_forest.fit(X_scaled, rf_labels)
        if progress_callback: progress_callback(85, "Random Forest trained.") # Changed from 70
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict anomalies using both models."""
        if self.feature_columns is None:
            raise ValueError("Model must be trained before prediction")
            
        X_scaled = self.scaler.transform(X[self.feature_columns])
        
        # Isolation Forest predictions
        if_scores = -self.isolation_forest.score_samples(X_scaled)
        if_predictions = (if_scores > np.percentile(if_scores, (1 - self.contamination) * 100)).astype(int) # Use contamination
        
        # LSTM Autoencoder predictions (only if trained)
        lstm_scores = np.zeros(len(X)) # Initialize with zeros
        lstm_predictions = np.zeros(len(X), dtype=int) # Initialize with zeros
        
        if self.lstm_autoencoder is not None and len(X_scaled) >= self.sequence_length:
            sequences = self._create_sequences(X_scaled)
            reconstructed = self.lstm_autoencoder.predict(sequences, verbose=0)
            
            # Calculate reconstruction error (MSE)
            mse = np.mean(np.power(sequences - reconstructed, 2), axis=(1, 2))
            
            # Map MSE scores back to original indices in X
            # The LSTM prediction covers X[self.sequence_length-1:]
            lstm_scores[self.sequence_length-1:] = mse
            
            # Define LSTM anomaly threshold (e.g., top 10% reconstruction error)
            lstm_threshold = np.percentile(lstm_scores[self.sequence_length-1:], (1 - self.contamination) * 100) # Use contamination
            lstm_predictions[self.sequence_length-1:] = (mse > lstm_threshold).astype(int)

        # Random Forest predictions (using features, not scores from other models directly)
        rf_predictions = self.random_forest.predict(X_scaled)
        
        # Combine predictions: Anomaly if IF OR LSTM OR RF predicts anomaly
        # Give more weight to IF and LSTM as primary anomaly detectors
        combined_predictions = np.maximum.reduce([if_predictions, lstm_predictions, rf_predictions])

        # A combined anomaly score could be an average or max of normalized scores
        # For simplicity, let's use IF scores as the primary score for plots for now, 
        # as LSTM scores are reconstruction error and need more careful normalization relative to IF scores.
        return combined_predictions, if_scores # Returning if_scores for visualization as it's consistent
    
    def save(self, path: str) -> None:
        """Save the model components."""
        model_data = {
            'scaler': self.scaler,
            'isolation_forest': self.isolation_forest,
            'random_forest': self.random_forest,
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'contamination': self.contamination, # Save contamination
            'lstm_autoencoder_config': self.lstm_autoencoder.get_config() if self.lstm_autoencoder else None,
            'lstm_autoencoder_weights': self.lstm_autoencoder.get_weights() if self.lstm_autoencoder else None
        }
        joblib.dump(model_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'HybridAnomalyDetector':
        """Load a saved model."""
        model_data = joblib.load(path)
        detector = cls(
            sequence_length=model_data['sequence_length'],
            lstm_units=model_data['lstm_units'],
            contamination=model_data['contamination'] # Load contamination
        )
        detector.scaler = model_data['scaler']
        detector.isolation_forest = model_data['isolation_forest']
        detector.random_forest = model_data['random_forest']
        detector.feature_columns = model_data['feature_columns']
        
        if model_data['lstm_autoencoder_config']:
            from tensorflow.keras.models import Model, model_from_config
            lstm_model = model_from_config(model_data['lstm_autoencoder_config'])
            lstm_model.set_weights(model_data['lstm_autoencoder_weights'])
            detector.lstm_autoencoder = lstm_model
        
        return detector

if __name__ == "__main__":
    # Example usage
    from data_generator import IAMLogGenerator
    from feature_engineering import IAMFeatureEngineer
    
    # Generate and prepare data
    generator = IAMLogGenerator()
    df = generator.generate_dataset(n_events=1000, anomaly_ratio=0.1)
    
    # Extract features
    engineer = IAMFeatureEngineer()
    df_features = engineer.extract_features(df)
    
    # Train model
    detector = HybridAnomalyDetector()
    detector.fit(df_features, engineer.get_feature_columns())
    
    # Make predictions
    predictions, scores = detector.predict(df_features)
    
    print("Anomaly Detection Results:")
    print(f"Number of anomalies detected: {predictions.sum()}")
    print(f"True anomalies in data: {df['is_anomaly'].sum()}")
    
    # Calculate accuracy
    accuracy = (predictions == df['is_anomaly']).mean()
    print(f"Accuracy: {accuracy:.2f}") 