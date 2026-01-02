import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Add, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
import joblib
import matplotlib.pyplot as plt
import pickle
import os
import json
from datetime import datetime
import logging

# === CONFIGURATION ===
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONFIG = {
    "app_name": "VitaMind AI",
    "version": "v11-production",
    "model_name": "calorie_predictor_v11_prod",
    "random_seed": 42,
    "test_size": 0.15,
    "val_size": 0.15,
    "n_synthetic_samples": 5000,
    "epochs": 300,
    "batch_size": 64,
    "initial_lr": 0.001,
    "patience": 30,
    "min_lr": 1e-6
}

# === DIRECTORIES ===
for dir_name in ["models", "logs", "plots", "data"]:
    os.makedirs(dir_name, exist_ok=True)

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === CALORIE CALCULATION ===
def calculate_calorie_goal(activity_level, weight, height, age, gender, 
                          sleep_hours, stress_level, mood, steps=5000, heart_rate=70):
    """
    Enhanced calorie calculation using Harris-Benedict + activity multipliers
    """
    # Basal Metabolic Rate (Mifflin-St Jeor Equation)
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == 'M' else -161)
    
    # Activity multipliers
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Active': 1.55,
        'Very Active': 1.725,
        'Athlete': 1.9
    }
    
    # Total Daily Energy Expenditure
    tdee = bmr * activity_factors.get(activity_level, 1.2)
    
    # Step bonus (40 cal per 1000 steps above baseline)
    step_bonus = max(0, (steps - 5000) * 0.04)
    
    # Heart rate adjustment (higher HR = more active)
    hr_bonus = max(0, (heart_rate - 70) * 0.5)
    
    # Sleep adjustment (poor sleep reduces metabolism)
    sleep_adjustment = 1 - 0.02 * max(0, 8 - sleep_hours)
    
    # Stress adjustment (stress increases cortisol)
    stress_adjustment = 1 + 0.03 * (stress_level / 10)
    
    # Mood adjustment
    mood_factors = {
        'happy': 1.05,
        'neutral': 1.0,
        'sad': 0.95,
        'anxious': 0.97
    }
    mood_adjustment = mood_factors.get(mood, 1.0)
    
    final_calories = (tdee + step_bonus + hr_bonus) * sleep_adjustment * stress_adjustment * mood_adjustment
    
    return round(final_calories, 2)

# === DATA GENERATION ===
def generate_synthetic_data(n_samples):
    """
    Generate realistic synthetic health data with correlations
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    np.random.seed(CONFIG['random_seed'])
    
    # Base demographics
    ages = np.random.randint(18, 80, n_samples)
    genders = np.random.choice(['M', 'F'], n_samples)
    
    # Correlated features
    activity_levels = np.random.choice(
        ['Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Athlete'],
        n_samples,
        p=[0.25, 0.30, 0.25, 0.15, 0.05]
    )
    
    # Steps correlate with activity level
    activity_step_map = {
        'Sedentary': (2000, 5000),
        'Lightly Active': (5000, 8000),
        'Active': (8000, 12000),
        'Very Active': (12000, 18000),
        'Athlete': (15000, 25000)
    }
    steps = np.array([
        np.random.randint(*activity_step_map[act]) 
        for act in activity_levels
    ])
    
    # Heart rate correlates with activity and age
    base_hr = 60 + (ages - 40) * 0.2
    hr_activity_boost = np.array([
        {'Sedentary': 0, 'Lightly Active': 5, 'Active': 10, 
         'Very Active': 15, 'Athlete': 20}[act]
        for act in activity_levels
    ])
    heart_rates = np.clip(base_hr + hr_activity_boost + np.random.normal(0, 5, n_samples), 50, 120)
    
    # Weight and height with realistic correlations
    heights = np.random.normal(170, 10, n_samples)
    heights = np.where(genders == 'M', heights + 8, heights)  # Men taller on average
    heights = np.clip(heights, 150, 200)
    
    bmi_targets = np.random.normal(25, 4, n_samples)
    weights = (bmi_targets * (heights / 100) ** 2)
    weights = np.clip(weights, 50, 120)
    
    # Sleep correlates with stress
    stress_levels = np.random.randint(1, 11, n_samples)
    sleep_hours = 8 - (stress_levels - 5) * 0.3 + np.random.normal(0, 0.5, n_samples)
    sleep_hours = np.clip(sleep_hours, 4, 10)
    
    # Mood distribution
    moods = np.random.choice(
        ['happy', 'neutral', 'sad', 'anxious'],
        n_samples,
        p=[0.30, 0.45, 0.15, 0.10]
    )
    
    data = pd.DataFrame({
        'age': ages,
        'gender': genders,
        'height': heights,
        'weight': weights,
        'activity_level': activity_levels,
        'steps': steps.astype(int),
        'heart_rate': heart_rates.astype(int),
        'sleep_hours': sleep_hours,
        'stress_level': stress_levels,
        'mood': moods
    })
    
    # Calculate target calories
    data['calories'] = data.apply(
        lambda row: calculate_calorie_goal(
            row['activity_level'], row['weight'], row['height'], 
            row['age'], row['gender'], row['sleep_hours'], 
            row['stress_level'], row['mood'], row['steps'], row['heart_rate']
        ),
        axis=1
    )
    
    logger.info(f"✓ Generated {len(data)} samples")
    return data

# === DATA VALIDATION & CLEANING ===
def validate_and_clean_data(data):
    """
    Validate data ranges and remove outliers
    """
    logger.info("Validating and cleaning data...")
    
    initial_count = len(data)
    
    # Define valid ranges
    validations = {
        'age': (18, 100),
        'weight': (40, 150),
        'height': (140, 220),
        'heart_rate': (50, 120),
        'sleep_hours': (3, 12),
        'stress_level': (1, 10),
        'steps': (0, 30000),
        'calories': (1000, 5000)
    }
    
    # Apply validations
    for col, (min_val, max_val) in validations.items():
        if col in data.columns:
            data = data[(data[col] >= min_val) & (data[col] <= max_val)]
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    # Remove rows with missing values
    data = data.dropna()
    
    logger.info(f"✓ Cleaned data: {initial_count} → {len(data)} samples ({len(data)/initial_count*100:.1f}% retained)")
    
    return data

# === FEATURE ENGINEERING ===
def engineer_features(data):
    """
    Create additional features for better predictions
    """
    logger.info("Engineering features...")
    
    # BMI
    data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
    
    # Age groups
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 100], labels=['young', 'middle', 'senior', 'elderly'])
    
    # Sleep quality (binary)
    data['good_sleep'] = (data['sleep_hours'] >= 7).astype(int)
    
    # High stress (binary)
    data['high_stress'] = (data['stress_level'] >= 7).astype(int)
    
    # Activity score (composite)
    activity_scores = {'Sedentary': 1, 'Lightly Active': 2, 'Active': 3, 'Very Active': 4, 'Athlete': 5}
    data['activity_score'] = data['activity_level'].map(activity_scores)
    
    # Steps category
    data['steps_category'] = pd.cut(data['steps'], bins=[0, 5000, 10000, 15000, 30000], 
                                    labels=['low', 'moderate', 'high', 'very_high'])
    
    logger.info(f"✓ Engineered {5} new features")
    return data

# === ENCODING ===
def encode_categorical_features(data):
    """
    Encode categorical variables
    """
    logger.info("Encoding categorical features...")
    
    encoders = {}
    
    # Label encoding for ordinal features
    categorical_cols = ['activity_level', 'gender', 'mood', 'age_group', 'steps_category']
    
    for col in categorical_cols:
        if col in data.columns:
            encoder = LabelEncoder()
            data[f'{col}_encoded'] = encoder.fit_transform(data[col])
            encoders[col] = encoder
    
    logger.info(f"✓ Encoded {len(categorical_cols)} categorical features")
    return data, encoders

# === MODEL BUILDING ===
def build_model(input_dim):
    """
    Build enhanced neural network with residual connections
    """
    logger.info(f"Building model with input dimension: {input_dim}")
    
    inputs = Input(shape=(input_dim,), name='input_layer')
    
    # First block
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.3, name='dropout_1')(x)
    
    # Second block with residual
    residual_1 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='residual_1')(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_3')(x)
    x = Add(name='add_1')([x, residual_1])
    
    # Third block
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense_4')(x)
    x = BatchNormalization(name='bn_3')(x)
    x = Dropout(0.2, name='dropout_3')(x)
    
    # Output layer
    outputs = Dense(1, name='output_layer')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='VitaMindAI_CaloriePredictor')
    
    logger.info(f"✓ Model built with {model.count_params():,} parameters")
    return model

# === TRAINING ===
def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train model with callbacks and logging
    """
    logger.info("Starting model training...")
    
    # Compile model
    optimizer = Adam(learning_rate=CONFIG['initial_lr'])
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG['patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=CONFIG['min_lr'],
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"models/{CONFIG['model_name']}_best.keras",
            monitor='val_mae',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    logger.info("✓ Training completed")
    return history

# === EVALUATION ===
def evaluate_model(model, X_test, y_test, scaler):
    """
    Comprehensive model evaluation
    """
    logger.info("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    # Metrics
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    
    metrics = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2_score': float(r2),
        'mean_error': float(np.mean(y_pred - y_test)),
        'std_error': float(np.std(y_pred - y_test))
    }
    
    logger.info(f"✓ Test MAE: {mae:.2f} kcal")
    logger.info(f"✓ Test RMSE: {rmse:.2f} kcal")
    logger.info(f"✓ Test MAPE: {mape:.2f}%")
    logger.info(f"✓ R² Score: {r2:.4f}")
    
    return metrics, y_pred

# === VISUALIZATION ===
def plot_results(history, y_test, y_pred, metrics):
    """
    Generate comprehensive training and evaluation plots
    """
    logger.info("Generating plots...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Training history
    plt.subplot(2, 3, 1)
    plt.plot(history.history['loss'], label='Train Loss', alpha=0.8)
    plt.plot(history.history['val_loss'], label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training & Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['mae'], label='Train MAE', alpha=0.8)
    plt.plot(history.history['val_mae'], label='Val MAE', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('MAE (kcal)')
    plt.title('Training & Validation MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Predictions vs Actual
    plt.subplot(2, 3, 3)
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Calories')
    plt.ylabel('Predicted Calories')
    plt.title('Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 3, 4)
    residuals = y_pred - y_test
    plt.scatter(y_pred, residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Calories')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 3, 5)
    plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error (kcal)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    
    # Metrics summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    metrics_text = f"""
    Model Performance Metrics
    ━━━━━━━━━━━━━━━━━━━━━━━
    MAE:  {metrics['mae']:.2f} kcal
    RMSE: {metrics['rmse']:.2f} kcal
    MAPE: {metrics['mape']:.2f}%
    R²:   {metrics['r2_score']:.4f}
    
    Mean Error: {metrics['mean_error']:.2f} kcal
    Std Error:  {metrics['std_error']:.2f} kcal
    """
    plt.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(f"plots/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", dpi=300)
    logger.info("✓ Plots saved")
    plt.close()

# === SAVE ARTIFACTS ===
def save_artifacts(model, scaler, encoders, metrics, feature_names):
    """
    Save all model artifacts and metadata
    """
    logger.info("Saving model artifacts...")
    
    # Save model
    model.save(f"models/{CONFIG['model_name']}.keras")
    
    # Save scaler
    joblib.dump(scaler, f"models/{CONFIG['model_name']}_scaler.joblib")
    
    # Save encoders
    joblib.dump(encoders, f"models/{CONFIG['model_name']}_encoders.joblib")
    
    # Save metadata
    metadata = {
        'config': CONFIG,
        'metrics': metrics,
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__,
        'model_summary': []
    }
    
    # Capture model summary
    model.summary(print_fn=lambda x: metadata['model_summary'].append(x))
    
    with open(f"models/{CONFIG['model_name']}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ All artifacts saved to models/ directory")

# === MAIN EXECUTION ===
def main():
    """
    Main training pipeline
    """
    logger.info("=" * 60)
    logger.info(f"🚀 {CONFIG['app_name']} {CONFIG['version']} - Training Pipeline")
    logger.info("=" * 60)
    
    # 1. Generate data
    data = generate_synthetic_data(CONFIG['n_synthetic_samples'])
    
    # 2. Validate and clean
    data = validate_and_clean_data(data)
    
    # 3. Engineer features
    data = engineer_features(data)
    
    # 4. Encode categorical features
    data, encoders = encode_categorical_features(data)
    
    # 5. Prepare features and target
    feature_columns = [
        'age', 'weight', 'height', 'steps', 'heart_rate', 
        'sleep_hours', 'stress_level', 'bmi',
        'activity_level_encoded', 'gender_encoded', 'mood_encoded',
        'good_sleep', 'high_stress', 'activity_score'
    ]
    
    X = data[feature_columns].values
    y = data['calories'].values
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target vector shape: {y.shape}")
    
    # 6. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 7. Split data (train/val/test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_seed']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=CONFIG['val_size'] / (1 - CONFIG['test_size']),
        random_state=CONFIG['random_seed']
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # 8. Build model
    model = build_model(input_dim=X_train.shape[1])
    
    # 9. Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # 10. Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test, scaler)
    
    # 11. Plot results
    plot_results(history, y_test, y_pred, metrics)
    
    # 12. Save artifacts
    save_artifacts(model, scaler, encoders, metrics, feature_columns)
    
    logger.info("=" * 60)
    logger.info("✅ Training pipeline completed successfully!")
    logger.info(f"📊 Final Test MAE: {metrics['mae']:.2f} kcal")
    logger.info(f"📁 Model saved as: {CONFIG['model_name']}.keras")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()