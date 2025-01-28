import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

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

# Build and Train the Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),  # Adjusted for 5 features (including activity level)
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile and Train
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=100, verbose=0)

# Save the model
model.save('calorie_predictor_v4.h5')

print("Model trained and saved successfully with expanded dataset!")
