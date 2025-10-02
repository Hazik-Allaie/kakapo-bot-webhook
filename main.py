from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import google.generativeai as genai
import os

app = Flask(__name__)

# 🔑 Set up Gemini API key
genai.configure(api_key=os.getenv("AIzaSyBjUrMg_DwG93K7YkfSdCnbnRMKPI0yMLA"))

# System prompt to constrain the model to kakapo topics
KAKAPO_SYSTEM_PROMPT = """You are an expert chatbot specializing exclusively in kakapo (Strigops habroptilus), the flightless parrot native to New Zealand.

Your role:
- Answer ONLY questions related to kakapo, including their biology, behavior, habitat, conservation status, breeding programs, history, and related topics.
- If a question is NOT about kakapo, politely respond: "I'm sorry, I only have knowledge about kakapo, the endangered flightless parrot of New Zealand. Please ask me anything about kakapo!"
- Be informative, accurate, and enthusiastic about kakapo.
- Do not answer questions about other topics, even if asked politely or persistently.

User question: """

def is_kakapo_related(question):
    """
    Pre-filter to check if question is likely about kakapo.
    Uses Gemini to determine relevance.
    """
    check_prompt = f"""Is the following question related to kakapo (the New Zealand parrot)? 
    Consider it related if it mentions:
    - Kakapo directly
    - Strigops habroptilus (scientific name)
    - New Zealand parrots in context that could include kakapo
    
    Question: "{question}"
    
    Answer with ONLY "YES" or "NO"."""
    
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(check_prompt)
        return "YES" in response.text.upper()
    except Exception as e:
        # If check fails, let the main prompt handle it
        print(f"Relevance check error: {e}")
        return True

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Kakapo Expert Chatbot API",
        "endpoints": ["/ask", "/analyze-image"]
    })

@app.route("/ask", methods=["POST"])
def ask_gemini():
    """Handle text-based questions about kakapo"""
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        # Quick pre-filter (optional but helps reduce API costs)
        if not is_kakapo_related(question):
            return jsonify({
                "answer": "I'm sorry, I only have knowledge about kakapo, the endangered flightless parrot of New Zealand. Please ask me anything about kakapo!",
                "on_topic": False
            })
        
        # Generate response with system prompt
        model = genai.GenerativeModel("gemini-pro")
        full_prompt = KAKAPO_SYSTEM_PROMPT + question
        response = model.generate_content(full_prompt)
        
        return jsonify({
            "answer": response.text,
            "on_topic": True
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze images for kakapo identification or features"""
    try:
        data = request.get_json()
        img_b64 = data.get("image", "")
        question = data.get("question", "Is this a kakapo?")
        
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode Base64 → bytes for Gemini
        img_bytes = base64.b64decode(img_b64)
        
        # Use Gemini Vision for kakapo-specific image analysis
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Create image part for Gemini
        image_part = {
            "mime_type": "image/jpeg",
            "data": img_bytes
        }
        
        prompt = f"""{KAKAPO_SYSTEM_PROMPT}
        
Image analysis request: {question}

If the image is not related to kakapo, respond: "I can only analyze images related to kakapo. Please share a kakapo image!"
"""
        
        response = model.generate_content([prompt, image_part])
        
        # Optional: Also do OpenCV processing for edge detection
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        opencv_analysis = None
        if img is not None:
            edges = cv2.Canny(img, 100, 200)
            edge_count = np.sum(edges > 0)
            opencv_analysis = {
                "edges_detected": int(edge_count),
                "image_shape": img.shape[:2]
            }
        
        return jsonify({
            "answer": response.text,
            "opencv_analysis": opencv_analysis
        })
        
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({"status": "healthy"}), 200

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__== "__main__":
    # Check if API key is set
    if not os.getenv("AIzaSyBjUrMg_DwG93K7YkfSdCnbnRMKPI0yMLA"):
        print("⚠  WARNING: GEMINI_API_KEY not set!")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
