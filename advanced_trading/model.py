"""
model.py
LSTM and deep learning models for the Advanced Stock Trading System.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Optional, Dict

class AdvancedLSTMModel:
    """
    High-performance LSTM model with attention mechanism for stock prediction.
    """
    def __init__(self, input_shape: Tuple[int, int], strategy: tf.distribute.Strategy):
        self.input_shape = input_shape
        self.strategy = strategy
        self.model: Optional[Model] = None
        self.history = None

    def build_model(self) -> Model:
        """Builds the LSTM model architecture."""
        with self.strategy.scope():
            inputs = Input(shape=self.input_shape)
            x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(inputs)
            x = LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(x)
            x = LSTM(32, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(x)
            x = Dense(50, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(25, activation='relu')(x)
            x = Dropout(0.2)(x)
            outputs = Dense(1, activation='linear')(x)
            model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
            self.model = model
            return model

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Trains the model with advanced callbacks."""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Makes predictions using the trained model."""
        return self.model.predict(X)

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray, scaler) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        Evaluates the model and returns metrics, predicted prices, and actual prices.
        """
        predictions = self.model.predict(X_test)
        pred_prices = scaler.inverse_transform(predictions)
        actual_prices = scaler.inverse_transform(y_test)
        mse = mean_squared_error(actual_prices, pred_prices)
        mae = mean_absolute_error(actual_prices, pred_prices)
        rmse = np.sqrt(mse)
        actual_direction = np.diff(actual_prices.flatten()) > 0
        pred_direction = np.diff(pred_prices.flatten()) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Directional_Accuracy': directional_accuracy
        }
        return metrics, pred_prices, actual_prices 