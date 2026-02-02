from flask import Flask, request, jsonify
import base64
import tempfile
import os
import librosa
import numpy as np

app = Flask(__name__)

SUPPORTED_LANGUAGES = ["tamil", "english", "hindi", "malayalam", "telugu"]

API_KEY = "YOUR_SECRET_API_KEY"   # change before submission


def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    features = np.hstack([mfcc_mean, spectral_centroid, spectral_bandwidth])
    return features


def predict_voice(features):
    """
    Placeholder ML logic.
    Replace this with a trained ML/DL model.
    """

    score = np.tanh(np.mean(features) / 100)

    confidence = abs(score)

    if score > 0:
        label = "AI_GENERATED"
    else:
        label = "HUMAN"

    return label, round(float(confidence), 3)


@app.route("/detect", methods=["POST"])
def detect_voice():
    # Authentication
    api_key = request.headers.get("X-API-KEY")
    if api_key != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    audio_base64 = data.get("audio")
    language = data.get("language", "").lower()

    if language not in SUPPORTED_LANGUAGES:
        return jsonify({"error": "Unsupported language"}), 400

    if not audio_base64:
        return jsonify({"error": "Audio missing"}), 400

    try:
        audio_bytes = base64.b64decode(audio_base64)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(audio_bytes)
            audio_path = tmp.name

        features = extract_features(audio_path)
        label, confidence = predict_voice(features)

        os.remove(audio_path)

        return jsonify({
            "language": language,
            "result": label,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
