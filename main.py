from flask import Flask, request, jsonify
import requests
import logging
from urllib.parse import quote
import os
from functools import lru_cache
import re
from textblob import TextBlob

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
WIKIPEDIA_API_BASE = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 5
HOST = os.getenv('FLASK_HOST', '0.0.0.0')
PORT = int(os.getenv('FLASK_PORT', 5000))

HEADERS = {
    "User-Agent": "KakapoChatBot/1.0 (https://example.com; contact@example.com)"
}

def clean_query(user_query: str) -> str:
    """Clean and correct user query for Wikipedia search."""
    query = user_query.lower().strip()
    
    # Remove common question words and phrases
    fillers = [
        "how many", "how much", "what is", "what are", "who is", "who are",
        "tell me about", "tell me something about", "give me information about",
        "explain", "describe", "define", "information about", "details about",
        "are left in", "are there in", "capital of"
    ]
    for f in fillers:
        query = query.replace(f, "")
    
    # Remove question words
    query = re.sub(r'\b(the|world|in|of|a|an)\b', '', query)
    
    # Remove punctuation
    query = re.sub(r"[^\w\s]", "", query)
    
    # Remove extra spaces
    query = " ".join(query.split())
    
    # Spell correction
    if query:
        try:
            corrected = str(TextBlob(query).correct())
        except:
            corrected = query
    else:
        corrected = query
    
    return corrected.strip()

def search_wikipedia(search_term: str):
    """Search Wikipedia and return the best matching page title."""
    try:
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': search_term,
            'format': 'json',
            'srlimit': 1  # Get only the best result
        }
        
        response = requests.get(
            WIKIPEDIA_API_BASE,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get('query', {}).get('search', [])
            
            if search_results:
                # Return the title of the best match
                return search_results[0]['title']
        
        return None
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return None

def get_wikipedia_extract(page_title: str):
    """Get the extract/summary for a specific Wikipedia page."""
    try:
        params = {
            'action': 'query',
            'prop': 'extracts',
            'exintro': True,  # Only the intro section
            'explaintext': True,  # Plain text (no HTML)
            'titles': page_title,
            'format': 'json'
        }
        
        response = requests.get(
            WIKIPEDIA_API_BASE,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            
            # Get the first (and only) page
            for page_id, page_data in pages.items():
                if 'extract' in page_data:
                    extract = page_data['extract']
                    # Limit to first 3 sentences for concise response
                    sentences = extract.split('. ')
                    if len(sentences) > 3:
                        return '. '.join(sentences[:3]) + '.'
                    return extract
        
        return None
        
    except Exception as e:
        logger.error(f"Extract error: {str(e)}")
        return None

@lru_cache(maxsize=128)
def get_wikipedia_summary(query):
    """
    Main function to get Wikipedia summary with search + extract.
    """
    if not query or not query.strip():
        return "Please provide a valid search query."
    
    # Clean the query
    cleaned_query = clean_query(query)
    
    if not cleaned_query:
        cleaned_query = query  # Use original if cleaning removed everything
    
    logger.info(f"Original query: '{query}' -> Cleaned: '{cleaned_query}'")
    
    try:
        # Step 1: Search for the best matching page
        page_title = search_wikipedia(cleaned_query)
        
        if not page_title:
            logger.warning(f"No Wikipedia page found for: {cleaned_query}")
            return "Sorry, I couldn't find a Wikipedia page for that topic."
        
        logger.info(f"Found Wikipedia page: {page_title}")
        
        # Step 2: Get the extract from the page
        extract = get_wikipedia_extract(page_title)
        
        if extract:
            logger.info(f"Successfully retrieved summary for: {page_title}")
            return extract
        else:
            logger.warning(f"No extract found for page: {page_title}")
            return "Sorry, I couldn't retrieve detailed information on that topic."
    
    except requests.exceptions.Timeout:
        logger.error(f"Timeout while fetching: {cleaned_query}")
        return "Sorry, the request took too long. Please try again."
    
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error while fetching: {cleaned_query}")
        return "Sorry, I couldn't connect to Wikipedia right now."
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return "Sorry, something went wrong. Please try again."

@app.route("/webhook", methods=["POST"])
def webhook():
    """Dialogflow webhook endpoint."""
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

@app.route("/test", methods=["GET"])
def test_query():
    """Test endpoint to verify Wikipedia API is working."""
    test_query = request.args.get('q', 'Kakapo')
    result = get_wikipedia_summary(test_query)
    return jsonify({
        "query": test_query,
        "result": result
    })

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