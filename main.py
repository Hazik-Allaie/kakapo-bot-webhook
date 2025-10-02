from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import google.generativeai as genai
import os

app = Flask(__name__)

# 🔑 Configure Gemini API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# System prompt (restricted to kakapo)
KAKAPO_SYSTEM_PROMPT = """You are an expert chatbot specializing exclusively in kakapo (Strigops habroptilus), 
the flightless parrot native to New Zealand.

Rules:
- Answer ONLY questions about kakapo (biology, habitat, conservation, etc.)
- If not about kakapo → say: "I'm sorry, I only have knowledge about kakapo, the endangered flightless parrot of New Zealand."
"""

# -------------------------
# ROUTES
# -------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Kakapo Expert Chatbot API",
        "endpoints": ["/ask", "/analyze-image", "/webhook"]
    })


@app.route("/ask", methods=["POST"])
def ask_gemini():
    """Direct text Q&A"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        model = genai.GenerativeModel("models/gemini-1.5-pro")
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\nUser question: " + question
        response = model.generate_content(full_prompt)

        return jsonify({"answer": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze kakapo-related images"""
    try:
        data = request.get_json()
        img_b64 = data.get("image", "")
        question = data.get("question", "Is this a kakapo?")

        if not img_b64:
            return jsonify({"error": "No image provided"}), 400

        img_bytes = base64.b64decode(img_b64)

        # Gemini Vision
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        response = model.generate_content([KAKAPO_SYSTEM_PROMPT + "\n" + question, image_part])

        # OpenCV Edge Detection
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        opencv_analysis = None
        if img is not None:
            edges = cv2.Canny(img, 100, 200)
            edge_count = int(np.sum(edges > 0))
            opencv_analysis = {"edges_detected": edge_count, "image_shape": img.shape[:2]}

        return jsonify({"answer": response.text, "opencv_analysis": opencv_analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/webhook", methods=["POST"])
def webhook():
    """Dialogflow fulfillment webhook"""
    try:
        req = request.get_json(force=True)
        query = req.get("queryResult", {}).get("queryText", "")

        if not query:
            return jsonify({"fulfillmentText": "No query received."})

        model = genai.GenerativeModel("models/gemini-1.5-pro")
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\nUser question: " + query
        response = model.generate_content(full_prompt)

        return jsonify({"fulfillmentText": response.text})
    except Exception as e:
        return jsonify({"fulfillmentText": f"Error: {str(e)}"})


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200


# -------------------------
# MAIN ENTRY
# -------------------------
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠ WARNING: GEMINI_API_KEY not set!")
    app.run(host="0.0.0.0", port=5000, debug=False)
