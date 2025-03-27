from flask import Flask, request, render_template_string, jsonify
from score import score
import joblib

# Initialize the Flask application
app = Flask("Model API")

# Load the pre-trained model
model = joblib.load("./../Assignment-2/mlruns/108559129931990894/affe3afbb88d4478959b5404ac51289e/artifacts/svc/model.pkl")

# Read the HTML template from the file
with open("app_template.html", "r") as file:
    html_template = file.read()

# Define the home route to render the HTML template
@app.route("/")
def home():
    return render_template_string(html_template, prediction=None, propensity=None)

# Define the predict route to handle form submissions
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text")
    threshold = request.form.get("threshold", "0.5")
    try:
        threshold = float(threshold) if threshold.strip() else 0.5
        
        # Call the scoring function with the input text, model, and threshold
        prediction, propensity = score(text, model, threshold)
        
        # Check if the request is from an API client
        if request.headers.get("Accept") == "application/json":
            return jsonify({"prediction": prediction, "propensity": round(propensity, 4)})
        
        # Otherwise, render the HTML template
        return render_template_string(html_template, prediction=prediction, propensity=round(propensity, 4))
    except Exception as e:
        if request.headers.get("Accept") == "application/json":
            return jsonify({"error": str(e)}), 400
        return render_template_string(html_template, prediction="Error", propensity=str(e))

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
