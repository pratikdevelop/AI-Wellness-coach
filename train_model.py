# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # from sklearn.model_selection import train_test_split
# # from tensorflow.keras import regularizers
# # from tensorflow.keras.optimizers.schedules import ExponentialDecay
# # from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Add, Input
# # from tensorflow.keras.models import Model
# # import joblib
# # import matplotlib.pyplot as plt
# # from collections import defaultdict
# # import pickle
# # import os
# # import re
# # from datasets import load_dataset
# # from llama_cpp import Llama

# # # Disable oneDNN for reproducibility
# # os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # # Define the calorie goal calculation function
# # def calculate_calorie_goal(activity_level, weight, height, age, gender, sleep_hours, stress_level, mood):
# #     if gender == 'M':
# #         bmr = 10 * weight + 6.25 * height - 5 * age + 5
# #     else:  # 'F'
# #         bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
# #     activity_factors = {
# #         'Sedentary': 1.2,
# #         'Lightly Active': 1.375,
# #         'Active': 1.55,
# #         'Very Active': 1.725
# #     }
# #     base_tdee = bmr * activity_factors.get(activity_level, 1.2)
    
# #     sleep_adjustment = 1 - (0.02 * (8 - sleep_hours))
# #     stress_adjustment = 1 + (0.03 * (stress_level / 10))
    
# #     mood_adjustments = {
# #         'happy': 1.05,
# #         'neutral': 1.0,
# #         'sad': 0.95,
# #         'anxious': 0.98
# #     }
# #     mood_factor = mood_adjustments.get(mood, 1.0)
    
# #     adjusted_tdee = base_tdee * sleep_adjustment * stress_adjustment * mood_factor
# #     return round(adjusted_tdee, 2)

# # # Q-Learning Agent for RL
# # class QLearningAgent:
# #     def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_table_path='q_table_train.pkl'):
# #         self.actions = actions
# #         self.lr = learning_rate
# #         self.gamma = discount_factor
# #         self.epsilon = exploration_rate
# #         self.q_table = defaultdict(lambda: np.zeros(len(actions)))
# #         self.q_table_path = q_table_path
# #         self.load_q_table()

# #     def discretize_state(self, state):
# #         mae = min(int(state['mae'] / 10), 10)
# #         epoch = min(int(state['epoch'] / 50), 10)
# #         stress_level = min(int(state['stress_level'] / 3), 3)
# #         mood_idx = state['mood_idx']
# #         return (mae, epoch, stress_level, mood_idx)

# #     def get_action(self, state):
# #         state_tuple = self.discretize_state(state)
# #         if np.random.rand() < self.epsilon:
# #             return np.random.choice(self.actions)
# #         else:
# #             q_values = self.q_table[state_tuple]
# #             return self.actions[np.argmax(q_values)]

# #     def update_q_table(self, state, action, reward, next_state):
# #         state_tuple = self.discretize_state(state)
# #         next_state_tuple = self.discretize_state(next_state)
# #         action_idx = self.actions.index(action)
        
# #         current_q = self.q_table[state_tuple][action_idx]
# #         next_max_q = np.max(self.q_table[next_state_tuple])
# #         new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
# #         self.q_table[state_tuple][action_idx] = new_q

# #     def save_q_table(self):
# #         with open(self.q_table_path, 'wb') as f:
# #             pickle.dump(dict(self.q_table), f)

# #     def load_q_table(self):
# #         if os.path.exists(self.q_table_path):
# #             with open(self.q_table_path, 'rb') as f:
# #                 self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), pickle.load(f))

# # # Load Mental-Health-FineTuned-Mistral-7B GGUF model
# # try:
# #     llm = Llama(
# #         model_path="i1-Q4_K_M.gguf",
# #         n_ctx=4096,
# #         n_threads=8,
# #         n_gpu_layers=35  # Adjust based on GPU availability
# #     )
# # except Exception as e:
# #     print(f"Error loading GGUF model: {str(e)}. Falling back to regex-based extraction.")
# #     llm = None

# # # Function to extract features from hari560/health dataset
# # def extract_features_from_health_dataset(dataset):
# #     data_list = []
    
# #     for item in dataset['train']:
# #         text = item['text']
# #         input_text = text.split('###Output')[0].replace('###Input :', '').strip()
        
# #         # Extract numerical features using regex
# #         age = re.search(r'(\d+)\s*y/o', input_text)
# #         gender = re.search(r'\((M|F)\)', input_text)
# #         height = re.search(r"(\d+'\d+\")", input_text)
# #         weight = re.search(r'(\d+)\s*lbs', input_text)
# #         heart_rate = re.search(r'HR.*?(\d+)', input_text)
        
# #         # Initialize defaults
# #         entry = {
# #             'age': None,
# #             'gender': None,
# #             'height': None,
# #             'weight': None,
# #             'heart_rate': None,
# #             'steps': 5000,
# #             'sleep_hours': 7.0,
# #             'stress_level': 5,
# #             'activity_level': 'Sedentary',
# #             'mood': 'neutral'
# #         }
        
# #         # Assign extracted values
# #         if age:
# #             entry['age'] = int(age.group(1))
# #         if gender:
# #             entry['gender'] = gender.group(1)
# #         if height:
# #             feet, inches = re.match(r"(\d+)'(\d+)", height.group(1)).groups()
# #             entry['height'] = float(feet) * 30.48 + float(inches) * 2.54
# #         if weight:
# #             entry['weight'] = float(weight.group(1)) * 0.453592
# #         if heart_rate:
# #             entry['heart_rate'] = int(heart_rate.group(1))
        
# #         # Use Mistral model for mood/stress inference if available
# #         if llm:
# #             prompt = f"""<s>[INST] Based on the following symptom description, classify the person's mood as 'happy', 'neutral', 'sad', or 'anxious', and estimate their stress level on a scale of 1 to 10. Provide the response in the format:
# #             Mood: <mood>
# #             Stress Level: <stress_level>
# #             Description: {input_text} [/INST]"""
# #             try:
# #                 output = llm(prompt, max_tokens=100, stop=["</s>"], echo=False)
# #                 response = output['choices'][0]['text'].strip()
# #                 mood_match = re.search(r'Mood:\s*(happy|neutral|sad|anxious)', response, re.IGNORECASE)
# #                 stress_match = re.search(r'Stress Level:\s*(\d+)', response)
# #                 if mood_match:
# #                     entry['mood'] = mood_match.group(1).lower()
# #                 if stress_match:
# #                     stress_level = int(stress_match.group(1))
# #                     if 1 <= stress_level <= 10:
# #                         entry['stress_level'] = stress_level
# #             except Exception as e:
# #                 print(f"Error processing text with Mistral: {str(e)}. Using fallback.")
        
# #         # Fallback to regex-based mood/stress inference
# #         if entry['mood'] == 'neutral' or entry['stress_level'] == 5:
# #             symptoms = input_text.lower()
# #             if any(word in symptoms for word in ['light headed', 'dizzy', 'breathless', 'lousy']):
# #                 entry['mood'] = 'sad'
# #                 entry['stress_level'] = 7
# #             if 'anxious' in symptoms or 'nervous' in symptoms:
# #                 entry['mood'] = 'anxious'
# #                 entry['stress_level'] = 8
        
# #         # Skip if critical features are missing
# #         if all(entry[key] is not None for key in ['age', 'gender', 'height', 'weight', 'heart_rate']):
# #             data_list.append(entry)
    
# #     return pd.DataFrame(data_list)

# # # Load and process hari560/health dataset
# # try:
# #     ds = load_dataset("hari560/health")
# #     health_data = extract_features_from_health_dataset(ds)
# # except Exception as e:
# #     print(f"Error loading health dataset: {str(e)}")
# #     health_data = pd.DataFrame()

# # # Generate synthetic dataset
# # np.random.seed(42)
# # n_samples = 200
# # synthetic_data = pd.DataFrame({
# #     'age': np.random.randint(18, 80, n_samples),
# #     'steps': np.random.randint(2000, 25000, n_samples),
# #     'weight': np.random.uniform(50, 120, n_samples),
# #     'height': np.random.uniform(150, 200, n_samples),
# #     'activity_level': np.random.choice(['Sedentary', 'Lightly Active', 'Active', 'Very Active'], n_samples),
# #     'gender': np.random.choice(['M', 'F'], n_samples),
# #     'heart_rate': np.random.randint(60, 100, n_samples),
# #     'sleep_hours': np.random.uniform(4, 10, n_samples),
# #     'stress_level': np.random.randint(1, 11, n_samples),
# #     'mood': np.random.choice(['happy', 'neutral', 'sad', 'anxious'], n_samples)
# # })

# # # Combine datasets
# # if not health_data.empty:
# #     health_data = health_data[synthetic_data.columns]
# #     data = pd.concat([synthetic_data, health_data], ignore_index=True)
# # else:
# #     data = synthetic_data

# # # Generate calorie targets
# # data['calories'] = data.apply(
# #     lambda row: calculate_calorie_goal(row['activity_level'], row['weight'], row['height'], row['age'],
# #                                        row['gender'], row['sleep_hours'], row['stress_level'], row['mood']),
# #     axis=1
# # )

# # # Encode categorical variables
# # label_encoder_activity = LabelEncoder()
# # data['activity_level_encoded'] = label_encoder_activity.fit_transform(data['activity_level'])
# # label_encoder_gender = LabelEncoder()
# # data['gender_encoded'] = label_encoder_gender.fit_transform(data['gender'])
# # label_encoder_mood = LabelEncoder()
# # data['mood_encoded'] = label_encoder_mood.fit_transform(data['mood'])

# # # Prepare features (X) and target (y)
# # X = data[['age', 'steps', 'weight', 'height', 'activity_level_encoded', 'gender_encoded',
# #           'heart_rate', 'sleep_hours', 'stress_level', 'mood_encoded']].values
# # y = data['calories'].values

# # # Scale the features
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # # Save scaler and encoders
# # joblib.dump(scaler, 'scaler.joblib')
# # joblib.dump(label_encoder_activity, 'label_encoder_activity.joblib')
# # joblib.dump(label_encoder_gender, 'label_encoder_gender.joblib')
# # joblib.dump(label_encoder_mood, 'label_encoder_mood.joblib')

# # # Split the data
# # X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # # Define the model
# # inputs = Input(shape=(10,))
# # x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.02))(inputs)
# # x = BatchNormalization()(x)
# # x = Dropout(0.4)(x)

# # residual = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
# # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
# # x = BatchNormalization()(x)
# # x = Dropout(0.4)(x)
# # x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
# # x = Add()([x, residual])

# # x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
# # outputs = Dense(1)(x)

# # model = Model(inputs=inputs, outputs=outputs)

# # # RL Agent to tune training
# # actions = ['increase_lr', 'decrease_lr', 'increase_calories', 'decrease_calories']
# # rl_agent = QLearningAgent(actions=actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)

# # # Compile model
# # initial_lr = tf.Variable(0.0005, trainable=False)
# # lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=10000, decay_rate=0.95)
# # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# # model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# # # Training with RL
# # epochs = 500
# # batch_size = 32
# # history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}

# # for epoch in range(epochs):
# #     hist = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
# #     history['loss'].append(hist.history['loss'][0])
# #     history['val_loss'].append(hist.history['val_loss'][0])
# #     history['mae'].append(hist.history['mae'][0])
# #     history['val_mae'].append(hist.history['val_mae'][0])
    
# #     state = {
# #         'mae': history['val_mae'][-1],
# #         'epoch': epoch,
# #         'stress_level': np.mean(data['stress_level']),
# #         'mood_idx': int(np.mean(data['mood_encoded']))
# #     }
    
# #     action = rl_agent.get_action(state)
    
# #     if action == 'increase_lr':
# #         initial_lr.assign(min(initial_lr * 1.1, 0.01))
# #     elif action == 'decrease_lr':
# #         initial_lr.assign(max(initial_lr * 0.9, 0.0001))
# #     elif action == 'increase_calories':
# #         y_train *= 1.05
# #         y_test *= 1.05
# #     elif action == 'decrease_calories':
# #         y_train *= 0.95
# #         y_test *= 0.95
    
# #     y_pred = model.predict(X_test, verbose=0)
# #     mae = np.mean(np.abs(y_pred.flatten() - y_test))
# #     stress_factor = 1 - (np.mean(data['stress_level']) / 10)
# #     mood_factor = 1 + (0.05 * (np.mean(data['mood_encoded']) - 1))
# #     reward = -mae * stress_factor * mood_factor
    
# #     next_state = {
# #         'mae': mae,
# #         'epoch': epoch + 1,
# #         'stress_level': np.mean(data['stress_level']),
# #         'mood_idx': int(np.mean(data['mood_encoded']))
# #     }
    
# #     rl_agent.update_q_table(state, action, reward, next_state)
    
# #     print(f"Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Val Loss: {history['val_loss'][-1]:.4f} - MAE: {mae:.4f} - Action: {action}")
    
# #     if epoch > 50 and history['val_loss'][-1] <= min(history['val_loss'][:-50]):
# #         print(f"Early stopping at epoch {epoch + 1}")
# #         break

# # # Save the Q-table
# # rl_agent.save_q_table()

# # # Evaluate the model
# # test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
# # print(f"Test Loss (Mean Squared Error): {test_loss}")
# # print(f"Test Mean Absolute Error: {test_mae}")

# # # Save the model
# # model.save('calorie_predictor_v9.keras')

# # # # Plot and save training data history (training loss and MAE only)
# # # plt.figure(figsize=(12, 5))

# # # # Subplot 1: Training Loss
# # # plt.subplot(1, 2, 1)
# # # plt.plot(history['loss'], label='Training Loss', color='blue')
# # # plt.title('Training Loss Over Epochs')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Loss (MSE)')
# # # plt.legend()
# # # plt.grid(True)

# # # # Subplot 2: Training MAE
# # # plt.subplot(1, 2, 2)
# # # plt.plot(history['mae'], label='Training MAE', color='green')
# # # plt.title('Training MAE Over Epochs')
# # # plt.xlabel('Epoch')
# # # plt.ylabel('Mean Absolute Error')
# # # plt.legend()
# # # plt.grid(True)

# # # plt.tight_layout()
# # # plt.savefig('training_data_plot.png')
# # # plt.show()


# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import regularizers
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Add, Input
# from tensorflow.keras.models import Model
# import joblib
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import pickle
# import os
# import re
# from datasets import load_dataset
# from llama_cpp import Llama

# # Disable oneDNN for reproducibility
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# # Define the calorie goal calculation function
# def calculate_calorie_goal(activity_level, weight, height, age, gender, sleep_hours, stress_level, mood):
#     if gender == 'M':
#         bmr = 10 * weight + 6.25 * height - 5 * age + 5
#     else:  # 'F'
#         bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
#     activity_factors = {
#         'Sedentary': 1.2,
#         'Lightly Active': 1.375,
#         'Active': 1.55,
#         'Very Active': 1.725
#     }
#     base_tdee = bmr * activity_factors.get(activity_level, 1.2)
    
#     sleep_adjustment = 1 - (0.02 * (8 - sleep_hours))
#     stress_adjustment = 1 + (0.03 * (stress_level / 10))
    
#     mood_adjustments = {
#         'happy': 1.05,
#         'neutral': 1.0,
#         'sad': 0.95,
#         'anxious': 0.98
#     }
#     mood_factor = mood_adjustments.get(mood, 1.0)
    
#     adjusted_tdee = base_tdee * sleep_adjustment * stress_adjustment * mood_factor
#     return round(adjusted_tdee, 2)

# # Q-Learning Agent for RL
# class QLearningAgent:
#     def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_table_path='q_table_train.pkl'):
#         self.actions = actions
#         self.lr = learning_rate
#         self.gamma = discount_factor
#         self.epsilon = exploration_rate
#         self.q_table = defaultdict(lambda: np.zeros(len(actions)))
#         self.q_table_path = q_table_path
#         self.load_q_table()

#     def discretize_state(self, state):
#         mae = min(int(state['mae'] / 10), 10)
#         epoch = min(int(state['epoch'] / 50), 10)
#         stress_level = min(int(state['stress_level'] / 3), 3)
#         mood_idx = state['mood_idx']
#         return (mae, epoch, stress_level, mood_idx)

#     def get_action(self, state):
#         state_tuple = self.discretize_state(state)
#         if np.random.rand() < self.epsilon:
#             return np.random.choice(self.actions)
#         else:
#             q_values = self.q_table[state_tuple]
#             return self.actions[np.argmax(q_values)]

#     def update_q_table(self, state, action, reward, next_state):
#         state_tuple = self.discretize_state(state)
#         next_state_tuple = self.discretize_state(next_state)
#         action_idx = self.actions.index(action)
        
#         current_q = self.q_table[state_tuple][action_idx]
#         next_max_q = np.max(self.q_table[next_state_tuple])
#         new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
#         self.q_table[state_tuple][action_idx] = new_q

#     def save_q_table(self):
#         with open(self.q_table_path, 'wb') as f:
#             pickle.dump(dict(self.q_table), f)

#     def load_q_table(self):
#         if os.path.exists(self.q_table_path):
#             with open(self.q_table_path, 'rb') as f:
#                 self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), pickle.load(f))

# # Load Mental-Health-FineTuned-Mistral-7B GGUF model
# try:
#     llm = Llama(
#         model_path="i1-Q4_K_M.gguf",
#         n_ctx=4096,
#         n_threads=8,
#         n_gpu_layers=35  # Adjust based on GPU availability
#     )
# except Exception as e:
#     print(f"Error loading GGUF model: {str(e)}. Falling back to regex-based extraction.")
#     llm = None

# # Function to extract features from hari560/health dataset
# def extract_features_from_health_dataset(dataset):
#     data_list = []
    
#     for item in dataset['train']:
#         text = item['text']
#         input_text = text.split('###Output')[0].replace('###Input :', '').strip()
        
#         # Extract numerical features using regex
#         age = re.search(r'(\d+)\s*y/o', input_text)
#         gender = re.search(r'\((M|F)\)', input_text)
#         height = re.search(r"(\d+'\d+\")", input_text)
#         weight = re.search(r'(\d+)\s*lbs', input_text)
#         heart_rate = re.search(r'HR.*?(\d+)', input_text)
        
#         # Initialize defaults
#         entry = {
#             'age': None,
#             'gender': None,
#             'height': None,
#             'weight': None,
#             'heart_rate': None,
#             'steps': 5000,
#             'sleep_hours': 7.0,
#             'stress_level': 5,
#             'activity_level': 'Sedentary',
#             'mood': 'neutral'
#         }
        
#         # Assign extracted values
#         if age:
#             entry['age'] = int(age.group(1))
#         if gender:
#             entry['gender'] = gender.group(1)
#         if height:
#             feet, inches = re.match(r"(\d+)'(\d+)", height.group(1)).groups()
#             entry['height'] = float(feet) * 30.48 + float(inches) * 2.54
#         if weight:
#             entry['weight'] = float(weight.group(1)) * 0.453592
#         if heart_rate:
#             entry['heart_rate'] = int(heart_rate.group(1))
        
#         # Use Mistral model for mood/stress inference if available
#         if llm:
#             prompt = f"""<s>[INST] Based on the following symptom description, classify the person's mood as 'happy', 'neutral', 'sad', or 'anxious', and estimate their stress level on a scale of 1 to 10. Provide the response in the format:
#             Mood: <mood>
#             Stress Level: <stress_level>
#             Description: {input_text} [/INST]"""
#             try:
#                 output = llm(prompt, max_tokens=100, stop=["</s>"], echo=False)
#                 response = output['choices'][0]['text'].strip()
#                 mood_match = re.search(r'Mood:\s*(happy|neutral|sad|anxious)', response, re.IGNORECASE)
#                 stress_match = re.search(r'Stress Level:\s*(\d+)', response)
#                 if mood_match:
#                     entry['mood'] = mood_match.group(1).lower()
#                 if stress_match:
#                     stress_level = int(stress_match.group(1))
#                     if 1 <= stress_level <= 10:
#                         entry['stress_level'] = stress_level
#             except Exception as e:
#                 print(f"Error processing text with Mistral: {str(e)}. Using fallback.")
        
#         # Fallback to regex-based mood/stress inference
#         if entry['mood'] == 'neutral' or entry['stress_level'] == 5:
#             symptoms = input_text.lower()
#             if any(word in symptoms for word in ['light headed', 'dizzy', 'breathless', 'lousy']):
#                 entry['mood'] = 'sad'
#                 entry['stress_level'] = 7
#             if 'anxious' in symptoms or 'nervous' in symptoms:
#                 entry['mood'] = 'anxious'
#                 entry['stress_level'] = 8
        
#         # Skip if critical features are missing
#         if all(entry[key] is not None for key in ['age', 'gender', 'height', 'weight', 'heart_rate']):
#             data_list.append(entry)
    
#     return pd.DataFrame(data_list)

# # Load and process hari560/health dataset
# try:
#     ds = load_dataset("hari560/health")
#     health_data = extract_features_from_health_dataset(ds)
# except Exception as e:
#     print(f"Error loading health dataset: {str(e)}")
#     health_data = pd.DataFrame()

# # Generate synthetic dataset
# np.random.seed(42)
# n_samples = 200
# synthetic_data = pd.DataFrame({
#     'age': np.random.randint(18, 80, n_samples),
#     'steps': np.random.randint(2000, 25000, n_samples),
#     'weight': np.random.uniform(50, 120, n_samples),
#     'height': np.random.uniform(150, 200, n_samples),
#     'activity_level': np.random.choice(['Sedentary', 'Lightly Active', 'Active', 'Very Active'], n_samples),
#     'gender': np.random.choice(['M', 'F'], n_samples),
#     'heart_rate': np.random.randint(60, 100, n_samples),
#     'sleep_hours': np.random.uniform(4, 10, n_samples),
#     'stress_level': np.random.randint(1, 11, n_samples),
#     'mood': np.random.choice(['happy', 'neutral', 'sad', 'anxious'], n_samples)
# })

# # Combine datasets
# if not health_data.empty:
#     health_data = health_data[synthetic_data.columns]
#     data = pd.concat([synthetic_data, health_data], ignore_index=True)
# else:
#     data = synthetic_data

# # Data validation and outlier removal
# data = data[
#     (data['calories'].between(1000, 4000)) &  # Typical calorie range
#     (data['weight'].between(40, 150)) &
#     (data['height'].between(140, 220)) &
#     (data['heart_rate'].between(50, 120)) &
#     (data['sleep_hours'].between(3, 12)) &
#     (data['stress_level'].between(1, 10)) &
#     (data['age'].between(18, 100))
# ]
# data = data.dropna()

# # Generate calorie targets
# data['calories'] = data.apply(
#     lambda row: calculate_calorie_goal(row['activity_level'], row['weight'], row['height'], row['age'],
#                                        row['gender'], row['sleep_hours'], row['stress_level'], row['mood']),
#     axis=1
# )

# # Encode categorical variables
# label_encoder_activity = LabelEncoder()
# data['activity_level_encoded'] = label_encoder_activity.fit_transform(data['activity_level'])
# label_encoder_gender = LabelEncoder()
# data['gender_encoded'] = label_encoder_gender.fit_transform(data['gender'])
# label_encoder_mood = LabelEncoder()
# data['mood_encoded'] = label_encoder_mood.fit_transform(data['mood'])

# # Prepare features (X) and target (y)
# X = data[['age', 'steps', 'weight', 'height', 'activity_level_encoded', 'gender_encoded',
#           'heart_rate', 'sleep_hours', 'stress_level', 'mood_encoded']].values
# y = data['calories'].values

# # Scale the features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Save scaler and encoders
# joblib.dump(scaler, 'scaler.joblib')
# joblib.dump(label_encoder_activity, 'label_encoder_activity.joblib')
# joblib.dump(label_encoder_gender, 'label_encoder_gender.joblib')
# joblib.dump(label_encoder_mood, 'label_encoder_mood.joblib')

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Define the model with adjusted regularization
# inputs = Input(shape=(10,))
# x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
# x = BatchNormalization()(x)
# x = Dropout(0.3)(x)

# residual = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# x = BatchNormalization()(x)
# x = Dropout(0.3)(x)
# x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# x = Add()([x, residual])

# x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
# outputs = Dense(1)(x)

# model = Model(inputs=inputs, outputs=outputs)

# # RL Agent to tune training
# actions = ['increase_lr', 'decrease_lr', 'increase_calories', 'decrease_calories']
# rl_agent = QLearningAgent(actions=actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)

# # Compile model with adjusted learning rate
# initial_lr = tf.Variable(0.001, trainable=False)
# lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=10000, decay_rate=0.95)
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# # Training with RL
# epochs = 500
# batch_size = 32
# history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
# calorie_scale_count = 0  # Track calorie scaling actions

# for epoch in range(epochs):
#     hist = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
#     history['loss'].append(hist.history['loss'][0])
#     history['val_loss'].append(hist.history['val_loss'][0])
#     history['mae'].append(hist.history['mae'][0])
#     history['val_mae'].append(hist.history['val_mae'][0])
    
#     state = {
#         'mae': history['val_mae'][-1],
#         'epoch': epoch,
#         'stress_level': np.mean(data['stress_level']),
#         'mood_idx': int(np.mean(data['mood_encoded']))
#     }
    
#     action = rl_agent.get_action(state)
    
#     # Apply action with constraints
#     if action == 'increase_lr':
#         initial_lr.assign(min(initial_lr * 1.1, 0.01))
#     elif action == 'decrease_lr':
#         initial_lr.assign(max(initial_lr * 0.9, 0.0001))
#     elif action == 'increase_calories' and calorie_scale_count < 5:
#         y_train *= 1.02  # Reduced scaling factor
#         y_test *= 1.02
#         calorie_scale_count += 1
#     elif action == 'decrease_calories' and calorie_scale_count < 5:
#         y_train *= 0.98  # Reduced scaling factor
#         y_test *= 0.98
#         calorie_scale_count += 1
    
#     y_pred = model.predict(X_test, verbose=0)
#     mae = np.mean(np.abs(y_pred.flatten() - y_test))
#     stress_factor = 1 - (np.mean(data['stress_level']) / 10)
#     mood_factor = 1 + (0.05 * (np.mean(data['mood_encoded']) - 1))
#     reward = -mae * stress_factor * mood_factor
    
#     next_state = {
#         'mae': mae,
#         'epoch': epoch + 1,
#         'stress_level': np.mean(data['stress_level']),
#         'mood_idx': int(np.mean(data['mood_encoded']))
#     }
    
#     rl_agent.update_q_table(state, action, reward, next_state)
    
#     print(f"Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Val Loss: {history['val_loss'][-1]:.4f} - MAE: {mae:.4f} - Action: {action}")
    
#     if epoch > 50 and history['val_loss'][-1] <= min(history['val_loss'][:-50]):
#         print(f"Early stopping at epoch {epoch + 1}")
#         break

# # Save the Q-table
# rl_agent.save_q_table()

# # Evaluate the model
# test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Loss (Mean Squared Error): {test_loss}")
# print(f"Test Mean Absolute Error: {test_mae}")

# # Save the model
# model.save('calorie_predictor_v9.keras')

# # # Plot and save training data history (training loss and MAE only)
# # plt.figure(figsize=(12, 5))

# # # Subplot 1: Training Loss
# # plt.subplot(1, 2, 1)
# # plt.plot(history['loss'], label='Training Loss', color='blue')
# # plt.title('Training Loss Over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('Loss (MSE)')
# # plt.legend()
# # plt.grid(True)

# # # Subplot 2: Training MAE
# # plt.subplot(1, 2, 2)
# # plt.plot(history['mae'], label='Training MAE', color='green')
# # plt.title('Training MAE Over Epochs')
# # plt.xlabel('Epoch')
# # plt.ylabel('Mean Absolute Error')
# # plt.legend()
# # plt.grid(True)

# # plt.tight_layout()
# # plt.savefig('training_data_plot.png')
# # plt.show()



import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Add, Input
from tensorflow.keras.models import Model
import joblib
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import os
import re
from datasets import load_dataset
from llama_cpp import Llama

# Disable oneDNN for reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the calorie goal calculation function
def calculate_calorie_goal(activity_level, weight, height, age, gender, sleep_hours, stress_level, mood):
    if gender == 'M':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # 'F'
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    
    activity_factors = {
        'Sedentary': 1.2,
        'Lightly Active': 1.375,
        'Active': 1.55,
        'Very Active': 1.725
    }
    base_tdee = bmr * activity_factors.get(activity_level, 1.2)
    
    sleep_adjustment = 1 - (0.02 * (8 - sleep_hours))
    stress_adjustment = 1 + (0.03 * (stress_level / 10))
    
    mood_adjustments = {
        'happy': 1.05,
        'neutral': 1.0,
        'sad': 0.95,
        'anxious': 0.98
    }
    mood_factor = mood_adjustments.get(mood, 1.0)
    
    adjusted_tdee = base_tdee * sleep_adjustment * stress_adjustment * mood_factor
    return round(adjusted_tdee, 2)

# Q-Learning Agent for RL
class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1, q_table_path='q_table_train.pkl'):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.q_table_path = q_table_path
        self.load_q_table()

    def discretize_state(self, state):
        mae = min(int(state['mae'] / 10), 10)
        epoch = min(int(state['epoch'] / 50), 10)
        stress_level = min(int(state['stress_level'] / 3), 3)
        mood_idx = state['mood_idx']
        return (mae, epoch, stress_level, mood_idx)

    def get_action(self, state):
        state_tuple = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = self.q_table[state_tuple]
            return self.actions[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        state_tuple = self.discretize_state(state)
        next_state_tuple = self.discretize_state(next_state)
        action_idx = self.actions.index(action)
        
        current_q = self.q_table[state_tuple][action_idx]
        next_max_q = np.max(self.q_table[next_state_tuple])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_tuple][action_idx] = new_q

    def save_q_table(self):
        with open(self.q_table_path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), pickle.load(f))

# Load Mental-Health-FineTuned-Mistral-7B GGUF model
gguf_path = os.path.join(os.path.dirname(__file__), "i1-Q4_K_M.gguf")
try:
    if not os.path.exists(gguf_path):
        raise FileNotFoundError(f"GGUF model file not found at: {gguf_path}. Please download it using: huggingface-cli download mradermacher/Mental-Health-FineTuned-Mistral-7B-Instruct-v0.2-GGUF i1-Q4_K_M.gguf --local-dir .")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=35  # Adjust based on GPU availability
    )
except Exception as e:
    print(f"Error loading GGUF model: {str(e)}. Falling back to regex-based extraction.")
    llm = None

# Function to extract features from hari560/health dataset
def extract_features_from_health_dataset(dataset):
    data_list = []
    
    for item in dataset['train']:
        text = item['text']
        input_text = text.split('###Output')[0].replace('###Input :', '').strip()
        
        # Extract numerical features using regex
        age = re.search(r'(\d+)\s*y/o', input_text)
        gender = re.search(r'\((M|F)\)', input_text)
        height = re.search(r"(\d+'\d+\")", input_text)
        weight = re.search(r'(\d+)\s*lbs', input_text)
        heart_rate = re.search(r'HR.*?(\d+)', input_text)
        
        # Initialize defaults
        entry = {
            'age': None,
            'gender': None,
            'height': None,
            'weight': None,
            'heart_rate': None,
            'steps': 5000,
            'sleep_hours': 7.0,
            'stress_level': 5,
            'activity_level': 'Sedentary',
            'mood': 'neutral'
        }
        
        # Assign extracted values
        if age:
            entry['age'] = int(age.group(1))
        if gender:
            entry['gender'] = gender.group(1)
        if height:
            feet, inches = re.match(r"(\d+)'(\d+)", height.group(1)).groups()
            entry['height'] = float(feet) * 30.48 + float(inches) * 2.54
        if weight:
            entry['weight'] = float(weight.group(1)) * 0.453592
        if heart_rate:
            entry['heart_rate'] = int(heart_rate.group(1))
        
        # Use Mistral model for mood/stress inference if available
        if llm:
            prompt = f"""<s>[INST] Based on the following symptom description, classify the person's mood as 'happy', 'neutral', 'sad', or 'anxious', and estimate their stress level on a scale of 1 to 10. Provide the response in the format:
            Mood: <mood>
            Stress Level: <stress_level>
            Description: {input_text} [/INST]"""
            try:
                output = llm(prompt, max_tokens=100, stop=["</s>"], echo=False)
                response = output['choices'][0]['text'].strip()
                mood_match = re.search(r'Mood:\s*(happy|neutral|sad|anxious)', response, re.IGNORECASE)
                stress_match = re.search(r'Stress Level:\s*(\d+)', response)
                if mood_match:
                    entry['mood'] = mood_match.group(1).lower()
                if stress_match:
                    stress_level = int(stress_match.group(1))
                    if 1 <= stress_level <= 10:
                        entry['stress_level'] = stress_level
            except Exception as e:
                print(f"Error processing text with Mistral: {str(e)}. Using fallback.")
        
        # Fallback to regex-based mood/stress inference
        if entry['mood'] == 'neutral' or entry['stress_level'] == 5:
            symptoms = input_text.lower()
            if any(word in symptoms for word in ['light headed', 'dizzy', 'breathless', 'lousy']):
                entry['mood'] = 'sad'
                entry['stress_level'] = 7
            if 'anxious' in symptoms or 'nervous' in symptoms:
                entry['mood'] = 'anxious'
                entry['stress_level'] = 8
        
        # Skip if critical features are missing
        if all(entry[key] is not None for key in ['age', 'gender', 'height', 'weight', 'heart_rate']):
            data_list.append(entry)
    
    return pd.DataFrame(data_list)

# Load and process hari560/health dataset
try:
    ds = load_dataset("hari560/health")
    health_data = extract_features_from_health_dataset(ds)
except Exception as e:
    print(f"Error loading health dataset: {str(e)}")
    health_data = pd.DataFrame()

# Generate synthetic dataset
np.random.seed(42)
n_samples = 200
synthetic_data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'steps': np.random.randint(2000, 25000, n_samples),
    'weight': np.random.uniform(50, 120, n_samples),
    'height': np.random.uniform(150, 200, n_samples),
    'activity_level': np.random.choice(['Sedentary', 'Lightly Active', 'Active', 'Very Active'], n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'heart_rate': np.random.randint(60, 100, n_samples),
    'sleep_hours': np.random.uniform(4, 10, n_samples),
    'stress_level': np.random.randint(1, 11, n_samples),
    'mood': np.random.choice(['happy', 'neutral', 'sad', 'anxious'], n_samples)
})

# Combine datasets
if not health_data.empty:
    health_data = health_data[synthetic_data.columns]
    data = pd.concat([synthetic_data, health_data], ignore_index=True)
else:
    data = synthetic_data

# Initial data validation (before calorie generation)
data = data[
    (data['weight'].between(40, 150)) &
    (data['height'].between(140, 220)) &
    (data['heart_rate'].between(50, 120)) &
    (data['sleep_hours'].between(3, 12)) &
    (data['stress_level'].between(1, 10)) &
    (data['age'].between(18, 100))
]
data = data.dropna()

# Generate calorie targets
data['calories'] = data.apply(
    lambda row: calculate_calorie_goal(row['activity_level'], row['weight'], row['height'], row['age'],
                                       row['gender'], row['sleep_hours'], row['stress_level'], row['mood']),
    axis=1
)

# Final data validation (after calorie generation)
data = data[
    (data['calories'].between(1000, 4000))  # Typical calorie range
]
data = data.dropna()

# Encode categorical variables
label_encoder_activity = LabelEncoder()
data['activity_level_encoded'] = label_encoder_activity.fit_transform(data['activity_level'])
label_encoder_gender = LabelEncoder()
data['gender_encoded'] = label_encoder_gender.fit_transform(data['gender'])
label_encoder_mood = LabelEncoder()
data['mood_encoded'] = label_encoder_mood.fit_transform(data['mood'])

# Prepare features (X) and target (y)
X = data[['age', 'steps', 'weight', 'height', 'activity_level_encoded', 'gender_encoded',
          'heart_rate', 'sleep_hours', 'stress_level', 'mood_encoded']].values
y = data['calories'].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler and encoders
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_activity, 'label_encoder_activity.joblib')
joblib.dump(label_encoder_gender, 'label_encoder_gender.joblib')
joblib.dump(label_encoder_mood, 'label_encoder_mood.joblib')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model with adjusted regularization
inputs = Input(shape=(10,))
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

residual = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Add()([x, residual])

x = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# RL Agent to tune training
actions = ['increase_lr', 'decrease_lr', 'increase_calories', 'decrease_calories']
rl_agent = QLearningAgent(actions=actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1)

# Compile model with adjusted learning rate
initial_lr = tf.Variable(0.001, trainable=False)
lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=10000, decay_rate=0.95)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

# Training with RL
epochs = 500
batch_size = 32
history = {'loss': [], 'val_loss': [], 'mae': [], 'val_mae': []}
calorie_scale_count = 0  # Track calorie scaling actions

for epoch in range(epochs):
    hist = model.fit(X_train, y_train, epochs=1, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
    
    history['loss'].append(hist.history['loss'][0])
    history['val_loss'].append(hist.history['val_loss'][0])
    history['mae'].append(hist.history['mae'][0])
    history['val_mae'].append(hist.history['val_mae'][0])
    
    state = {
        'mae': history['val_mae'][-1],
        'epoch': epoch,
        'stress_level': np.mean(data['stress_level']),
        'mood_idx': int(np.mean(data['mood_encoded']))
    }
    
    action = rl_agent.get_action(state)
    
    # Apply action with constraints
    if action == 'increase_lr':
        initial_lr.assign(min(initial_lr * 1.1, 0.01))
    elif action == 'decrease_lr':
        initial_lr.assign(max(initial_lr * 0.9, 0.0001))
    elif action == 'increase_calories' and calorie_scale_count < 5:
        y_train *= 1.02  # Reduced scaling factor
        y_test *= 1.02
        calorie_scale_count += 1
    elif action == 'decrease_calories' and calorie_scale_count < 5:
        y_train *= 0.98  # Reduced scaling factor
        y_test *= 0.98
        calorie_scale_count += 1
    
    y_pred = model.predict(X_test, verbose=0)
    mae = np.mean(np.abs(y_pred.flatten() - y_test))
    stress_factor = 1 - (np.mean(data['stress_level']) / 10)
    mood_factor = 1 + (0.05 * (np.mean(data['mood_encoded']) - 1))
    reward = -mae * stress_factor * mood_factor
    
    next_state = {
        'mae': mae,
        'epoch': epoch + 1,
        'stress_level': np.mean(data['stress_level']),
        'mood_idx': int(np.mean(data['mood_encoded']))
    }
    
    rl_agent.update_q_table(state, action, reward, next_state)
    
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {history['loss'][-1]:.4f} - Val Loss: {history['val_loss'][-1]:.4f} - MAE: {mae:.4f} - Action: {action}")
    
    if epoch > 50 and history['val_loss'][-1] <= min(history['val_loss'][:-50]):
        print(f"Early stopping at epoch {epoch + 1}")
        break

# Save the Q-table
rl_agent.save_q_table()

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (Mean Squared Error): {test_loss}")
print(f"Test Mean Absolute Error: {test_mae}")

# Save the model
model.save('calorie_predictor_v9.keras')

# Plot and save training data history (training loss and MAE only)
plt.figure(figsize=(12, 5))

# Subplot 1: Training Loss
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss', color='blue')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Subplot 2: Training MAE
plt.subplot(1, 2, 2)
plt.plot(history['mae'], label='Training MAE', color='green')
plt.title('Training MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_data_plot.png')
plt.show()