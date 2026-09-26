"""
AI Prediction Models
Train models for production, temperature, energy, and failure risk prediction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, accuracy_score,
    precision_recall_fscore_support
)
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, List


def temporal_split(X, y, test_fraction=0.2):
    """Keep later observations for testing instead of shuffling time-series data."""
    split_index = max(1, int(len(X) * (1 - test_fraction)))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]


class ProductionPredictor:
    """Predict oil production based on CSS and SRP parameters"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'steam_volume', 'steam_pressure', 'injection_duration',
            'soak_time', 'spm', 'stroke_length', 'vfd_frequency',
            'reservoir_temperature', 'oil_viscosity'
        ]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train production prediction model"""
        
        # Prepare features and target
        X = df[self.feature_names].values
        y = df['actual_production'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = temporal_split(X_scaled, y)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model_type': 'XGBoost',
            'mae': round(mean_absolute_error(y_test, y_pred), 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 3),
            'samples_trained': len(X_train)
        }
    
    def predict(self, input_dict: Dict) -> float:
        """Predict production for given parameters"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Prepare input
        features = np.array([[
            input_dict.get('steam_volume', 80),
            input_dict.get('steam_pressure', 25),
            input_dict.get('injection_duration', 24),
            input_dict.get('soak_time', 36),
            input_dict.get('spm', 4.0),
            input_dict.get('stroke_length', 86),
            input_dict.get('vfd_frequency', 45),
            input_dict.get('reservoir_temperature', 50),
            input_dict.get('oil_viscosity', 500)
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        return max(0, float(prediction))
    
    def save(self, filename='production_model.pkl'):
        """Save model"""
        joblib.dump(self.model, filename)
        joblib.dump(self.scaler, filename.replace('.pkl', '_scaler.pkl'))
    
    def load(self, filename='production_model.pkl'):
        """Load model"""
        self.model = joblib.load(filename)
        self.scaler = joblib.load(filename.replace('.pkl', '_scaler.pkl'))


class TemperaturePredictor:
    """Predict reservoir temperature based on CSS parameters"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'steam_volume', 'steam_pressure', 'injection_duration',
            'soak_time', 'previous_temperature'
        ]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train temperature prediction model"""
        
        # Create previous temperature feature
        df['previous_temperature'] = df.groupby('well_id')['reservoir_temperature'].shift(1)
        df['previous_temperature'] = df['previous_temperature'].fillna(46.0)
        
        X = df[self.feature_names].values
        y = df['reservoir_temperature'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = temporal_split(X_scaled, y)
        
        # Train model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model_type': 'XGBoost',
            'mae': round(mean_absolute_error(y_test, y_pred), 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 3),
            'samples_trained': len(X_train)
        }
    
    def predict(self, input_dict: Dict) -> float:
        """Predict temperature"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        features = np.array([[
            input_dict.get('steam_volume', 80),
            input_dict.get('steam_pressure', 25),
            input_dict.get('injection_duration', 24),
            input_dict.get('soak_time', 36),
            input_dict.get('previous_temperature', 46)
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        return float(prediction)
    
    def save(self, filename='temperature_model.pkl'):
        joblib.dump(self.model, filename)
        joblib.dump(self.scaler, filename.replace('.pkl', '_scaler.pkl'))
    
    def load(self, filename='temperature_model.pkl'):
        self.model = joblib.load(filename)
        self.scaler = joblib.load(filename.replace('.pkl', '_scaler.pkl'))


class EnergyPredictor:
    """Predict energy consumption"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'spm', 'stroke_length', 'vfd_frequency', 'rod_load',
            'oil_viscosity', 'reservoir_temperature'
        ]
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train energy prediction model"""
        
        X = df[self.feature_names].values
        y = df['energy_consumption'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = temporal_split(X_scaled, y)
        
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return {
            'model_type': 'RandomForest',
            'mae': round(mean_absolute_error(y_test, y_pred), 3),
            'rmse': round(rmse, 3),
            'r2_score': round(r2, 3),
            'samples_trained': len(X_train)
        }
    
    def predict(self, input_dict: Dict) -> float:
        if self.model is None:
            raise ValueError("Model not trained")
        
        features = np.array([[
            input_dict.get('spm', 4.0),
            input_dict.get('stroke_length', 86),
            input_dict.get('vfd_frequency', 45),
            input_dict.get('rod_load', 75),
            input_dict.get('oil_viscosity', 500),
            input_dict.get('reservoir_temperature', 50)
        ]])
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)[0]
        
        return max(0, float(prediction))
    
    def save(self, filename='energy_model.pkl'):
        joblib.dump(self.model, filename)
        joblib.dump(self.scaler, filename.replace('.pkl', '_scaler.pkl'))
    
    def load(self, filename='energy_model.pkl'):
        self.model = joblib.load(filename)
        self.scaler = joblib.load(filename.replace('.pkl', '_scaler.pkl'))


class FailureRiskClassifier:
    """Classify failure risk (High/Medium/Low)"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.class_prior = np.array([1.0, 0.0, 0.0])
        self.feature_names = [
            'rod_load', 'spm', 'oil_viscosity', 'motor_current',
            'stroke_length', 'vfd_frequency'
        ]
        self.risk_labels = ['Low', 'Medium', 'High']
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train failure risk classifier"""
        
        # Create risk categories based on failure_risk score
        df['risk_category'] = pd.cut(
            df['failure_risk'],
            bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]
        )
        
        X = df[self.feature_names].values
        y = df['risk_category'].fillna(0).values.astype(int)
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = temporal_split(X_scaled, y)
        self.class_prior = np.bincount(y_train, minlength=3).astype(float)
        self.class_prior /= max(self.class_prior.sum(), 1.0)
        
        base_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        class_counts = np.bincount(y_train)
        calibration_folds = min(3, int(class_counts[class_counts > 0].min())) if np.any(class_counts > 0) else 0
        self.model = (
            CalibratedClassifierCV(base_model, method='sigmoid', cv=calibration_folds)
            if calibration_folds >= 2 else base_model
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        return {
            'model_type': 'RandomForestClassifier',
            'accuracy': round(accuracy, 3),
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1_score': round(f1, 3),
            'samples_trained': len(X_train)
        }
    
    def predict(self, input_dict: Dict) -> Tuple[str, float]:
        """Predict failure risk and return risk level + probability"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        features = np.array([[
            input_dict.get('rod_load', 75),
            input_dict.get('spm', 4.0),
            input_dict.get('oil_viscosity', 500),
            input_dict.get('motor_current', 1000),
            input_dict.get('stroke_length', 86),
            input_dict.get('vfd_frequency', 45)
        ]])
        
        features_scaled = self.scaler.transform(features)
        risk_class = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        risk_label = self.risk_labels[risk_class]
        # Synthetic labels are highly separable; shrink confidence toward the
        # training prior until field outcomes provide calibration evidence.
        probability = min(
            0.95,
            float(0.85 * probabilities[risk_class] + 0.15 * self.class_prior[risk_class])
        )
        
        return risk_label, probability
    
    def save(self, filename='failure_risk_model.pkl'):
        joblib.dump(self.model, filename)
        joblib.dump(self.scaler, filename.replace('.pkl', '_scaler.pkl'))
        joblib.dump(self.class_prior, filename.replace('.pkl', '_prior.pkl'))
    
    def load(self, filename='failure_risk_model.pkl'):
        self.model = joblib.load(filename)
        self.scaler = joblib.load(filename.replace('.pkl', '_scaler.pkl'))
        prior_file = filename.replace('.pkl', '_prior.pkl')
        if os.path.exists(prior_file):
            self.class_prior = joblib.load(prior_file)


class ModelManager:
    """Manage all AI models"""
    
    def __init__(self):
        self.production_predictor = ProductionPredictor()
        self.temperature_predictor = TemperaturePredictor()
        self.energy_predictor = EnergyPredictor()
        self.failure_risk_classifier = FailureRiskClassifier()
    
    def train_all(self, df: pd.DataFrame) -> Dict:
        """Train all models"""
        print("Training AI models...")
        
        results = {
            'production_model': self.production_predictor.train(df),
            'temperature_model': self.temperature_predictor.train(df),
            'energy_model': self.energy_predictor.train(df),
            'failure_risk_model': self.failure_risk_classifier.train(df)
        }
        
        return results
    
    def predict_all(self, input_dict: Dict) -> Dict:
        """Generate all predictions for given parameters"""
        
        return {
            'production': round(self.production_predictor.predict(input_dict), 2),
            'temperature': round(self.temperature_predictor.predict(input_dict), 2),
            'energy': round(self.energy_predictor.predict(input_dict), 2),
            'failure_risk': self.failure_risk_classifier.predict(input_dict)
        }
    
    def save_all(self, model_dir: str = 'models'):
        """Save all models"""
        os.makedirs(model_dir, exist_ok=True)
        
        self.production_predictor.save(os.path.join(model_dir, 'production_model.pkl'))
        self.temperature_predictor.save(os.path.join(model_dir, 'temperature_model.pkl'))
        self.energy_predictor.save(os.path.join(model_dir, 'energy_model.pkl'))
        self.failure_risk_classifier.save(os.path.join(model_dir, 'failure_risk_model.pkl'))
        
        print(f"Models saved to {model_dir}")
    
    def load_all(self, model_dir: str = 'models'):
        """Load all models"""
        self.production_predictor.load(os.path.join(model_dir, 'production_model.pkl'))
        self.temperature_predictor.load(os.path.join(model_dir, 'temperature_model.pkl'))
        self.energy_predictor.load(os.path.join(model_dir, 'energy_model.pkl'))
        self.failure_risk_classifier.load(os.path.join(model_dir, 'failure_risk_model.pkl'))
        
        print(f"Models loaded from {model_dir}")


if __name__ == "__main__":
    # Load synthetic data
    df = pd.read_csv('data/synthetic_data.csv')
    
    # Train all models
    manager = ModelManager()
    results = manager.train_all(df)
    
    print("\n" + "="*80)
    print("MODEL TRAINING RESULTS")
    print("="*80)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"  {key:.<25} {value}")
    
    # Save models
    manager.save_all()
    
    # Test predictions
    print("\n" + "="*80)
    print("TEST PREDICTIONS")
    print("="*80)
    
    test_input = {
        'steam_volume': 80,
        'steam_pressure': 25,
        'injection_duration': 24,
        'soak_time': 36,
        'spm': 4.0,
        'stroke_length': 86,
        'vfd_frequency': 45,
        'rod_load': 75,
        'oil_viscosity': 500,
        'reservoir_temperature': 50,
        'motor_current': 1000,
        'previous_temperature': 46
    }
    
    predictions = manager.predict_all(test_input)
    
    print("\nInput Parameters:")
    for key, value in test_input.items():
        print(f"  {key:.<30} {value}")
    
    print("\nPredictions:")
    print(f"  Production (BOPD)          {predictions['production']}")
    print(f"  Temperature (°C)           {predictions['temperature']}")
    print(f"  Energy (kW)                {predictions['energy']}")
    print(f"  Failure Risk               {predictions['failure_risk'][0]} ({predictions['failure_risk'][1]:.2%})")
