# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf

# Enhanced Dataset
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'steps': [5000, 10000, 15000, 20000, 25000],
    'weight': [70, 75, 80, 85, 90],  # kg
    'height': [170, 175, 180, 185, 190],  # cm
    'activity_level': ['Sedentary', 'Lightly Active', 'Active', 'Very Active', 'Active'],
    'calories': [200, 300, 400, 500, 600]
})

# Prepare Dataset
X = data[['age', 'steps', 'weight', 'height']].values
y = data['calories'].values

# Build and Train the Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),  # Adjusted for 4 features
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, verbose=0)

# Save the model
model.save('calorie_predictor_v3.h5')

# Optionally, you can reload the model after training
# model = tf.keras.models.load_model('calorie_predictor_v3.h5')

print("Model trained and saved successfully!")
