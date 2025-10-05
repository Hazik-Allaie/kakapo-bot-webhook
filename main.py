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

# Safety settings for Gemini
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

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
        
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            safety_settings=SAFETY_SETTINGS
        )
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\nUser question: " + question
        response = model.generate_content(full_prompt)
        
        return jsonify({"answer": response.text})
    
    except Exception as e:
        print(f"Error in /ask: {str(e)}")
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
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            safety_settings=SAFETY_SETTINGS
        )
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
        print(f"Error in /analyze-image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """Dialogflow fulfillment webhook"""
    try:
        req = request.get_json(force=True)
        print(f"Received webhook request: {req}")  # Debug logging
        
        # Extract query from Dialogflow request
        query = req.get("queryResult", {}).get("queryText", "")
        
        if not query:
            return jsonify({
                "fulfillmentText": "No query received.",
                "source": "kakapo-chatbot"
            })
        
        # Generate response using Gemini
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            safety_settings=SAFETY_SETTINGS
        )
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\nUser question: " + query
        response = model.generate_content(full_prompt)
        
        # Format response for Dialogflow
        dialogflow_response = {
            "fulfillmentText": response.text,
            "source": "kakapo-chatbot"
        }
        
        # Optional: Add rich responses for Dialogflow Messenger
        # dialogflow_response["fulfillmentMessages"] = [
        #     {
        #         "text": {
        #             "text": [response.text]
        #         }
        #     }
        # ]
        
        return jsonify(dialogflow_response)
    
    except Exception as e:
        print(f"Error in /webhook: {str(e)}")
        return jsonify({
            "fulfillmentText": f"I apologize, but I encountered an error: {str(e)}",
            "source": "kakapo-chatbot"
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    return jsonify({
        "status": "healthy",
        "api_key_configured": api_key_set
    }), 200

# -------------------------
# MAIN ENTRY
# -------------------------
if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
    else:
        print("✅ GEMINI_API_KEY configured")
    
    app.run(host="0.0.0.0", port=5000, debug=True)  # Use debug=True for development
