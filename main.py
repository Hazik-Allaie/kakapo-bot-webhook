from flask import Flask, request, jsonify
import requests
import logging
from urllib.parse import quote
import os
from functools import lru_cache

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/api/rest_v1/page/summary/"
REQUEST_TIMEOUT = 5  # seconds
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', 5000))

# Add required User-Agent header for Wikipedia
HEADERS = {
    "User-Agent": "KakapoChatBot/1.0 (https://example.com; contact@example.com)"
}

# Cache Wikipedia results for 1 hour (maxsize=128 queries)
@lru_cache(maxsize=128)
def get_wikipedia_summary(query):
    """
    Fetch summary from Wikipedia API with error handling and caching.
    """
    if not query or not query.strip():
        return "Please provide a valid search query."
    
    try:
        encoded_query = quote(query.strip())
        url = f"{WIKIPEDIA_API_BASE}{encoded_query}"
        
        logger.info(f"Fetching Wikipedia summary for: {query}")
        
        # Make request with headers + timeout
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'extract' in data and data['extract']:
                logger.info(f"Successfully retrieved summary for: {query}")
                return data['extract']
            else:
                logger.warning(f"No extract found for: {query}")
                return "Sorry, I couldn't find detailed information on that topic."
        
        elif response.status_code == 404:
            logger.warning(f"Wikipedia page not found: {query}")
            return "Sorry, I couldn't find a Wikipedia page for that topic."
        
        else:
            logger.error(f"Wikipedia API error: {response.status_code}")
            return "Sorry, there was an issue retrieving information from Wikipedia."
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching: {query}")
        return "Sorry, the request took too long. Please try again."
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while fetching: {query}")
        return "Sorry, I couldn't connect to Wikipedia right now."
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return "Sorry, there was an error processing your request."
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "Sorry, something went wrong. Please try again."

@app.route("/webhook", methods=["POST"])
def webhook():
    """
    Dialogflow webhook endpoint.
    """
    try:
        req = request.get_json(force=True, silent=True)
        
        if not req:
            logger.error("Empty or invalid JSON request")
            return jsonify({
                "fulfillmentText": "Sorry, I received an invalid request."
            }), 400
        
        query_result = req.get("queryResult", {})
        user_query = query_result.get("queryText", "").strip()
        
        if not user_query:
            logger.warning("No query text found in request")
            return jsonify({
                "fulfillmentText": "I didn't understand your question. Please try again."
            })
        
        logger.info(f"Processing query: {user_query}")
        
        # Get Wikipedia summary
        answer = get_wikipedia_summary(user_query)
        
        return jsonify({
            "fulfillmentText": answer
        })
    
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        return jsonify({
            "fulfillmentText": "Sorry, there was an error processing your request."
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Wikipedia Webhook"
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": "Internal server error"
    }), 500

if __name__ == "__main__":
    logger.info(f"Starting Flask server on {HOST}:{PORT}")
    app.run(
        host=HOST,
        port=PORT,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    )
