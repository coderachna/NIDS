# Piece 1: Imports
from flask import Flask
import joblib
from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Piece 2: Load saved model, scaler, and label encoder
rf_model = joblib.load(r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\rf_model.pkl")
scaler = joblib.load(r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\scaler.pkl")
le = joblib.load(r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\label_encoder.pkl")

print("Model, scaler, and label encoder loaded successfully.")

# Piece 3: Rebuild the test set (same split as training)
DATA_PATH = r"C:\Users\rachn\OneDrive\Desktop\companiesresume\projects\NIDS-CICIDS2017\data\cleaned_data.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['Label'])
y = df['Label']

# Use the already-fitted label encoder (transform, NOT fit_transform)
y_encoded = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("X_test shape:", X_test.shape)

# Piece 4: Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return '''
    <h1>Network Intrusion Detection System</h1>
    <p>Click the button to classify a random network flow from the test set.</p>
    <button onclick="getPrediction()">Get Random Flow</button>
    <div id="result" style="margin-top:20px; font-size:18px;"></div>

    <hr style="margin:30px 0;">

    <h2>Batch Classification (CSV Upload)</h2>
    <p>Upload a CSV of network flows (same feature columns as the training data) to classify them all at once.</p>
    <form action="/batch_predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv" required>
        <button type="submit">Classify CSV</button>
    </form>

    <script>
    async function getPrediction() {
        const response = await fetch('/predict');
        const data = await response.json();
        document.getElementById('result').innerHTML = `
            <p><b>Predicted Class:</b> ${data.predicted_class}</p>
            <p><b>Confidence:</b> ${data.confidence}%</p>
            <p><b>Actual Class:</b> ${data.actual_class}</p>
            <p><b>Correct?</b> ${data.correct ? "✅ Yes" : "❌ No"}</p>
        `;
    }
    </script>
    '''
@app.route('/predict')
def predict():
    # Pick a random row from the test set
    random_idx = np.random.randint(0, len(X_test))
    row = X_test.iloc[[random_idx]]  # keep as DataFrame (2D) for scaler/model
    true_label_encoded = y_test[random_idx]

    # Scale it using the SAME scaler from training
    row_scaled = scaler.transform(row)

    # Predict class + get probability for confidence
    pred_encoded = rf_model.predict(row_scaled)[0]
    pred_proba = rf_model.predict_proba(row_scaled)[0]
    confidence = round(max(pred_proba) * 100, 2)

    # Decode numbers back into readable class names
    predicted_class = le.inverse_transform([pred_encoded])[0]
    actual_class = le.inverse_transform([true_label_encoded])[0]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "actual_class": actual_class,
        "correct": bool(pred_encoded == true_label_encoded)
    }
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    file = request.files['file']
    batch_df = pd.read_csv(file)

    # If the uploaded CSV happens to include a Label column, drop it before predicting
    has_labels = 'Label' in batch_df.columns
    if has_labels:
        true_labels = batch_df['Label']
        features = batch_df.drop(columns=['Label'])
    else:
        features = batch_df

    # Scale and predict
    features_scaled = scaler.transform(features)
    preds_encoded = rf_model.predict(features_scaled)
    preds_proba = rf_model.predict_proba(features_scaled)
    predicted_classes = le.inverse_transform(preds_encoded)
    confidences = [round(max(p) * 100, 2) for p in preds_proba]

    # Build results table
    results_df = features.copy()
    results_df['Predicted_Class'] = predicted_classes
    results_df['Confidence'] = confidences
    if has_labels:
        results_df['Actual_Class'] = true_labels.values

    # Summary counts per predicted class
    summary = results_df['Predicted_Class'].value_counts().to_dict()

    html = f"<h2>Batch Results</h2><p><b>Total rows classified:</b> {len(results_df)}</p>"
    html += "<p><b>Predicted class breakdown:</b></p><ul>"
    for cls, count in summary.items():
        html += f"<li>{cls}: {count}</li>"
    html += "</ul>"
    html += "<p><a href='/'>← Back</a></p>"
    html += results_df.head(50).to_html(index=False)

    return html
if __name__ == '__main__':
    app.run(debug=True)