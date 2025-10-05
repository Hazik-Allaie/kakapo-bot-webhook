from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import google.generativeai as genai
import os

app = Flask(__name__)

# 🔑 Configure Gemini API key from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)

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

# Model configuration with fallback options
def get_model(prefer_vision=False):
    """Get the best available Gemini model with fallback options"""
    
    # Try models in order of preference for free tier
    if prefer_vision:
        models_to_try = [
            "gemini-2.0-flash",           # Latest free tier with vision
            "gemini-1.5-flash",           # Stable free tier
            "gemini-1.5-flash-latest",    # Latest stable
        ]
    else:
        models_to_try = [
            "gemini-2.0-flash",           # Latest free tier
            "gemini-1.5-flash",           # Stable free tier  
            "gemini-1.5-flash-latest",    # Latest stable
        ]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name,
                safety_settings=SAFETY_SETTINGS
            )
            print(f"✅ Using model: {model_name}")
            return model
        except Exception as e:
            print(f"⚠️  Model {model_name} not available: {str(e)}")
            continue
    
    # If all fail, raise error
    raise Exception("No available Gemini models found. Please check your API key and quota.")

# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "service": "Kakapo Expert Chatbot API",
        "endpoints": ["/ask", "/analyze-image", "/webhook", "/list-models"]
    })

@app.route("/list-models", methods=["GET"])
def list_models():
    """List available Gemini models for debugging"""
    try:
        if not API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 500
        
        models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                models.append({
                    "name": m.name,
                    "display_name": m.display_name,
                })
        return jsonify({"available_models": models, "count": len(models)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask_gemini():
    """Direct text Q&A"""
    try:
        if not API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not configured"}), 500
        
        data = request.get_json()
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "No question provided"}), 400
        
        model = get_model(prefer_vision=False)
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\n\nUser question: " + question
        response = model.generate_content(full_prompt)
        
        return jsonify({"answer": response.text})
    
    except Exception as e:
        print(f"❌ Error in /ask: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze-image", methods=["POST"])
def analyze_image():
    """Analyze kakapo-related images"""
    try:
        if not API_KEY:
            return jsonify({"error": "GEMINI_API_KEY not configured"}), 500
        
        data = request.get_json()
        img_b64 = data.get("image", "")
        question = data.get("question", "Is this a kakapo?")
        
        if not img_b64:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode image
        img_bytes = base64.b64decode(img_b64)
        
        # Gemini Vision
        model = get_model(prefer_vision=True)
        image_part = {"mime_type": "image/jpeg", "data": img_bytes}
        prompt = KAKAPO_SYSTEM_PROMPT + "\n\n" + question
        response = model.generate_content([prompt, image_part])
        
        # OpenCV Edge Detection
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        opencv_analysis = None
        
        if img is not None:
            edges = cv2.Canny(img, 100, 200)
            edge_count = int(np.sum(edges > 0))
            opencv_analysis = {
                "edges_detected": edge_count, 
                "image_shape": list(img.shape[:2])
            }
        
        return jsonify({
            "answer": response.text, 
            "opencv_analysis": opencv_analysis
        })
    
    except Exception as e:
        print(f"❌ Error in /analyze-image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/webhook", methods=["POST"])
def webhook():
    """Dialogflow fulfillment webhook"""
    try:
        if not API_KEY:
            return jsonify({
                "fulfillmentText": "API key not configured. Please contact administrator.",
                "source": "kakapo-chatbot"
            })
        
        req = request.get_json(force=True)
        print(f"📨 Received webhook request: {req}")
        
        # Extract query from Dialogflow request
        query = req.get("queryResult", {}).get("queryText", "")
        
        if not query:
            return jsonify({
                "fulfillmentText": "No query received.",
                "source": "kakapo-chatbot"
            })
        
        # Generate response using Gemini
        model = get_model(prefer_vision=False)
        full_prompt = KAKAPO_SYSTEM_PROMPT + "\n\nUser question: " + query
        response = model.generate_content(full_prompt)
        
        # Format response for Dialogflow
        dialogflow_response = {
            "fulfillmentText": response.text,
            "source": "kakapo-chatbot"
        }
        
        print(f"✅ Sending response: {response.text[:100]}...")
        return jsonify(dialogflow_response)
    
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error in /webhook: {error_msg}")
        
        # User-friendly error messages
        if "404" in error_msg or "not found" in error_msg:
            user_message = "I'm having trouble connecting to my AI service. Please try again in a moment."
        elif "quota" in error_msg.lower():
            user_message = "I've reached my usage limit. Please try again later."
        elif "api key" in error_msg.lower():
            user_message = "There's a configuration issue. Please contact the administrator."
        else:
            user_message = "I encountered an error while processing your request. Please try again."
        
        return jsonify({
            "fulfillmentText": user_message,
            "source": "kakapo-chatbot"
        })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    api_key_set = bool(API_KEY)
    
    # Try to verify API access
    can_access_api = False
    model_info = "Not checked"
    
    if api_key_set:
        try:
            model = get_model(prefer_vision=False)
            can_access_api = True
            model_info = "API accessible"
        except Exception as e:
            model_info = f"API error: {str(e)[:100]}"
    
    return jsonify({
        "status": "healthy" if api_key_set else "warning",
        "api_key_configured": api_key_set,
        "api_accessible": can_access_api,
        "model_status": model_info
    }), 200

# -------------------------
# MAIN ENTRY
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🦜 KAKAPO CHATBOT API")
    print("="*50)
    
    if not API_KEY:
        print("⚠️  WARNING: GEMINI_API_KEY not set!")
        print("Set it with: export GEMINI_API_KEY='your-key-here'")
        print("Get your key at: https://aistudio.google.com/app/apikey")
    else:
        print("✅ GEMINI_API_KEY configured")
        
        # Try to list available models
        try:
            print("\n📋 Checking available models...")
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    model_name = m.name.replace('models/', '')
                    models.append(model_name)
            
            if models:
                print(f"✅ Found {len(models)} available models:")
                for model in models[:5]:  # Show first 5
                    print(f"   - {model}")
                if len(models) > 5:
                    print(f"   ... and {len(models) - 5} more")
            else:
                print("⚠️  No models found. Your API key may have issues.")
        except Exception as e:
            print(f"⚠️  Could not list models: {str(e)}")
    
    print("\n🚀 Starting server on http://0.0.0.0:5000")
    print("="*50 + "\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)
