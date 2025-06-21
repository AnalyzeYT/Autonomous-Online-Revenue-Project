"""
model.py
200% Advanced ML Model Management for the Advanced Stock Trading System.
Includes abstract base class, registry, hyperparameter tuning, explainability, ensembling, experiment tracking, and cloud save/load.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Optional, Dict, Any, List, Callable
from abc import ABC, abstractmethod

# Optional: MLflow/W&B integration (stub)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Set up logging for the module
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Abstract base class for all models
class BaseModel(ABC):
    """
    Abstract base class for all ML models in the trading system.
    """
    @abstractmethod
    def build_model(self):
        pass
    @abstractmethod
    def train(self, X_train, y_train, X_val, y_val, config: dict):
        pass
    @abstractmethod
    def predict(self, X):
        pass
    @abstractmethod
    def evaluate(self, X_test, y_test, scaler) -> Tuple[Dict, np.ndarray, np.ndarray]:
        pass
    @abstractmethod
    def save(self, path: str):
        pass
    @abstractmethod
    def load(self, path: str):
        pass

# Model registry for versioning and experiment tracking
class ModelRegistry:
    def __init__(self, registry_dir: str = ".model_registry"):
        self.registry_dir = registry_dir
        if not os.path.exists(registry_dir):
            os.makedirs(registry_dir)
    def save_model(self, model: BaseModel, name: str, version: str = "latest"):
        path = os.path.join(self.registry_dir, f"{name}_{version}.h5")
        model.save(path)
        logger.info(f"[REGISTRY] Saved model {name} version {version} to {path}")
    def load_model(self, model_class: Callable, name: str, version: str = "latest") -> BaseModel:
        path = os.path.join(self.registry_dir, f"{name}_{version}.h5")
        model = model_class()
        model.load(path)
        logger.info(f"[REGISTRY] Loaded model {name} version {version} from {path}")
        return model

# Advanced LSTM Model
class AdvancedLSTMModel(BaseModel):
    def __init__(self, input_shape: Tuple[int, int], config: dict = None):
        self.input_shape = input_shape
        self.config = config or {}
        self.model: Optional[Model] = None
        self.history = None
    def build_model(self):
        units1 = self.config.get('lstm_units1', 128)
        units2 = self.config.get('lstm_units2', 64)
        units3 = self.config.get('lstm_units3', 32)
        dropout = self.config.get('dropout', 0.2)
        dense1 = self.config.get('dense1', 50)
        dense2 = self.config.get('dense2', 25)
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            inputs = Input(shape=self.input_shape)
            x = LSTM(units1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(inputs)
            x = LSTM(units2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(x)
            x = LSTM(units3, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(x)
            x = Dense(dense1, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(dense2, activation='relu')(x)
            x = Dropout(dropout)(x)
            outputs = Dense(1, activation='linear')(x)
            self.model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001), beta_1=0.9, beta_2=0.999)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        return self.model
    def train(self, X_train, y_train, X_val, y_val, config: dict = None):
        config = config or self.config
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.get('patience', 15), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        if MLFLOW_AVAILABLE:
            mlflow.log_params(config)
            mlflow.log_metrics({'final_val_loss': self.history.history['val_loss'][-1]})
        return self.history
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test, scaler) -> Tuple[Dict, np.ndarray, np.ndarray]:
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
    def save(self, path: str):
        if self.model:
            self.model.save(path)
            logger.info(f"[MODEL] Model saved to {path}")
        else:
            logger.warning("[MODEL] No model to save.")
    def load(self, path: str):
        if os.path.exists(path):
            self.model = load_model(path)
            logger.info(f"[MODEL] Model loaded from {path}")
        else:
            logger.error(f"[MODEL] File {path} does not exist.")

# GRU Model (alternative)
class GRUModel(BaseModel):
    def __init__(self, input_shape: Tuple[int, int], config: dict = None):
        self.input_shape = input_shape
        self.config = config or {}
        self.model: Optional[Model] = None
        self.history = None
    def build_model(self):
        units1 = self.config.get('gru_units1', 128)
        units2 = self.config.get('gru_units2', 64)
        units3 = self.config.get('gru_units3', 32)
        dropout = self.config.get('dropout', 0.2)
        dense1 = self.config.get('dense1', 50)
        dense2 = self.config.get('dense2', 25)
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            inputs = Input(shape=self.input_shape)
            x = GRU(units1, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(inputs)
            x = GRU(units2, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)(x)
            x = GRU(units3, return_sequences=False, dropout=dropout, recurrent_dropout=dropout)(x)
            x = Dense(dense1, activation='relu')(x)
            x = Dropout(dropout)(x)
            x = Dense(dense2, activation='relu')(x)
            x = Dropout(dropout)(x)
            outputs = Dense(1, activation='linear')(x)
            self.model = Model(inputs=inputs, outputs=outputs)
            optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001), beta_1=0.9, beta_2=0.999)
            self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        return self.model
    def train(self, X_train, y_train, X_val, y_val, config: dict = None):
        config = config or self.config
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=config.get('patience', 15), restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001)
        ]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        if MLFLOW_AVAILABLE:
            mlflow.log_params(config)
            mlflow.log_metrics({'final_val_loss': self.history.history['val_loss'][-1]})
        return self.history
    def predict(self, X):
        return self.model.predict(X)
    def evaluate(self, X_test, y_test, scaler) -> Tuple[Dict, np.ndarray, np.ndarray]:
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
    def save(self, path: str):
        if self.model:
            self.model.save(path)
            logger.info(f"[MODEL] Model saved to {path}")
        else:
            logger.warning("[MODEL] No model to save.")
    def load(self, path: str):
        if os.path.exists(path):
            self.model = load_model(path)
            logger.info(f"[MODEL] Model loaded from {path}")
        else:
            logger.error(f"[MODEL] File {path} does not exist.")

# Ensembling utility
class ModelEnsembler:
    def __init__(self, models: List[BaseModel]):
        self.models = models
    def predict(self, X) -> np.ndarray:
        preds = [m.predict(X) for m in self.models]
        return np.mean(preds, axis=0)
    def evaluate(self, X_test, y_test, scaler) -> Tuple[Dict, np.ndarray, np.ndarray]:
        preds = self.predict(X_test)
        pred_prices = scaler.inverse_transform(preds)
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

# Hyperparameter tuning (Optuna stub)
def hyperparameter_tuning(model_class: Callable, X_train, y_train, X_val, y_val, param_space: dict, n_trials: int = 20):
    try:
        import optuna
    except ImportError:
        logger.warning("[TUNING] Optuna not installed. Skipping tuning.")
        return None
    def objective(trial):
        config = {k: trial.suggest_float(k, *v) if isinstance(v, tuple) else v for k, v in param_space.items()}
        model = model_class(config['input_shape'], config)
        model.build_model()
        model.train(X_train, y_train, X_val, y_val, config)
        val_loss = model.history.history['val_loss'][-1]
        return val_loss
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info(f"[TUNING] Best params: {study.best_params}")
    return study.best_params

# Model explainability (SHAP stub)
def explain_model(model: BaseModel, X_sample: np.ndarray):
    try:
        import shap
    except ImportError:
        logger.warning("[EXPLAIN] SHAP not installed. Skipping explainability.")
        return None
    explainer = shap.DeepExplainer(model.model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    logger.info("[EXPLAIN] SHAP values computed.")
    return shap_values

# Utility function for model selection
def get_model(model_type: str, input_shape: Tuple[int, int], config: dict = None) -> BaseModel:
    if model_type.lower() == 'lstm':
        return AdvancedLSTMModel(input_shape, config)
    elif model_type.lower() == 'gru':
        return GRUModel(input_shape, config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}") 