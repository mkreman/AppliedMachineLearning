from flask import Flask, request, render_template_string, jsonify
from score import score
import joblib

# Initialize the Flask application
app = Flask("Model API")

# Load the pre-trained model
model = joblib.load("./../Assignment-2/mlruns/108559129931990894/affe3afbb88d4478959b5404ac51289e/artifacts/svc/model.pkl")

@app.route('/', methods=['GET'])
def home():
    return "Flask API is running"

# Define the predict route to handle form submissions
@app.route("/score", methods=["POST"])
def score_endpoint():
    data = request.get_json()
    text = data.get('text')
    threshold = float(data.get('threshold', 0.5))
    try:
        # Call the scoring function with the input text, model, and threshold
        prediction, propensity = score(text, model, threshold)
        
        return jsonify({"prediction": prediction, "propensity": round(propensity, 4)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
