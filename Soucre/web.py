from flask import Flask, request, jsonify, render_template
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Attempt to load the pre-trained machine learning model
try:
    model = joblib.load('randomforestmodel.pkl')  # Ensure the filename matches the one you saved your model as
    # Optionally load definitions or mapping for Iris species if saved separately
    # definitions = joblib.load('definitions.pkl')  # Assuming you saved definitions as a separate file
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# For demonstration, these are assumed to be the classes corresponding to the model's predictions
definitions = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


@app.route('/')
def home():
    # Render the HTML template for the home page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on the HTML GUI
    '''
    if not model:
        return render_template('index.html', prediction_text="Model is not loaded. Please check the server setup.")

    try:
        # Extract input features from the form submission
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        scaler = joblib.load('scaler.pkl')
        features = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

        prediction = model.predict(features)
        output = definitions[prediction[0]]

        # Render the index HTML template with the prediction result
        prediction_text = f'Dự đoán Iris: {output}'
    except Exception as e:
        prediction_text = f"Lỗi dự đoán: {e}"

    return render_template('index.html', prediction_text=prediction_text)


# Run the Flask application
if __name__ == "__main__":
    app.run(debug=True)
