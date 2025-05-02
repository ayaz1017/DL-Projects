from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow import keras

app = Flask(__name__)

# Load the trained model
model = keras.models.load_model('model_ann.h5', custom_objects={'mse': keras.losses.MeanSquaredError()})
model.compile(optimizer="adam", loss="mse")

# Load the scaler
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Encoding for ocean proximity
ocean_mapping = {"inland": 1, "near ocean": 2, "near bay": 3, "island": 4}

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/house", methods=['POST', 'GET'])
def house():
    if request.method == 'POST':
        try:
            # Get form data
            longitude = float(request.form['longitude'])
            latitude = float(request.form['latitude'])
            houseage = float(request.form['houseage'])
            houserooms = float(request.form['houserooms'])
            totalbedrooms = float(request.form['totlabedrooms'])
            population = float(request.form['population'])
            households = float(request.form['households'])
            medianincome = float(request.form['medianincome'])
            oceanproximity = request.form['oceanproximity'].strip().lower()

            # Convert ocean proximity to numerical value
            oceanproximity = ocean_mapping.get(oceanproximity, 0)  # Default to 0 if not found

            # Prepare features array
            features = np.array([longitude, latitude, houseage, houserooms, totalbedrooms,
                                 population, households, medianincome, oceanproximity], dtype=float)

            # Apply scaling
            features_scaled = scaler.transform([features])

            # Predict house price
            price = model.predict(features_scaled)

            return render_template('index.html', result=round(price[0][0], 2))

        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/doc')
def doc():
    return render_template('doc.html')

@app.route('/ann')
def ann():
    return render_template('ann.html')

if __name__ == "__main__":
    app.run(debug=True)


