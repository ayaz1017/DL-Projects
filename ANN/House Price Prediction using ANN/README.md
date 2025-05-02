House Price Prediction using Artificial Neural Networks (ANN)
This project implements a house price prediction model using an Artificial Neural Network (ANN). The model is trained on housing data and aims to predict house prices based on various features.
📌 Features
Data preprocessing: Handling missing values and encoding categorical variables.
Feature scaling using MinMaxScaler.
ANN model built using Keras with Sequential API.
Training with early stopping to prevent overfitting.
Performance evaluation using metrics such as MAE, MSE, and R² score.
Model visualization with loss graphs and scatter plots.
Custom prediction function to estimate house prices based on user input.
🛠️ Technologies Used
Python
NumPy
Pandas
Matplotlib & Seaborn
Scikit-learn
Keras & TensorFlow
📂 Dataset
The dataset is assumed to be named housing.csv and should contain features such as:

longitude
latitude
housing_median_age
total_rooms
total_bedrooms
population
households
median_income
ocean_proximity
median_house_value (Target Variable)

📊 Model Architecture
Input Layer: Number of features in the dataset.
Hidden Layers:
Dense(1000, activation='relu') + Dropout(0.2)
Dense(500, activation='relu') + Dropout(0.2)
Dense(250, activation='relu')
Output Layer: Dense(1, activation='linear')
🎯 Results
The model's performance is evaluated using:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
R² Score
