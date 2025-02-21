import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Expanded Dataset with more diverse values
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 72, 29, 34, 40, 46, 60, 35, 22, 31, 48],
    'steps': [5000, 10000, 15000, 20000, 25000, 8000, 12000, 15000, 17000, 19000, 18000, 7000, 11000, 13000, 16000, 14000, 9000, 6000, 8500, 10000],
    'weight': [70, 75, 80, 85, 90, 65, 78, 82, 87, 90, 80, 70, 72, 85, 88, 92, 75, 68, 77, 79],  # kg
    'height': [170, 175, 180, 185, 190, 160, 165, 170, 175, 180, 185, 167, 172, 178, 182, 186, 170, 168, 173, 177],  # cm
    'activity_level': ['Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Active', 'Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Active', 'Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Active', 'Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Active'],
    'calories': [200, 300, 400, 500, 600, 250, 320, 400, 480, 550, 500, 220, 280, 350, 450, 550, 300, 210, 320, 400]
})

# Encode the 'activity_level' column
label_encoder = LabelEncoder()
data['activity_level_encoded'] = label_encoder.fit_transform(data['activity_level'])

# Prepare Dataset (including the encoded activity level)
X = data[['age', 'steps', 'weight', 'height', 'activity_level_encoded']].values
y = data['calories'].values

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Learning rate decay
lr_schedule = ExponentialDecay(initial_learning_rate=0.001, decay_steps=100000, decay_rate=0.96, staircase=True)

# Build and Train the Model with L2 Regularization, Dropout, and Learning Rate Scheduling
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,), 
                          kernel_regularizer=regularizers.l2(0.01)),  # L2 Regularization
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

# Compile the model with a learning rate scheduler
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss='mean_squared_error')

# EarlyStopping callback to stop training if validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=8, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])

# Save the trained model in Keras format
model.save('calorie_predictor.h5')

print("Model trained and saved successfully with improved features!")

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss (Mean Squared Error): {test_loss}")
