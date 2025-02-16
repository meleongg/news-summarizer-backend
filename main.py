import os
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from functools import lru_cache
import logging

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")
FRONTEND_FULL_URL = os.getenv("FRONTEND_FULL_URL")
LOCAL_FRONTEND_URL = os.getenv("LOCAL_FRONTEND_URL")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL")
GNEWS_API_URL = os.getenv("GNEWS_API_URL")
MAX_WORDS = int(os.getenv("MAX_WORDS"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD"))

app = FastAPI()

# CORS middleware
origins = [
    LOCAL_FRONTEND_URL,
    FRONTEND_URL,
    FRONTEND_FULL_URL,
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download data for Sentiment Analysis
nltk.download('vader_lexicon')
# Load NLP models
sentiment_analyzer = SentimentIntensityAnalyzer()

# URL validation function
def validate_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/fetch_news/")
async def fetch_news(request: Request, query: str, sort_by: str = "relevance", page_size: int = 5):
    """Fetch news articles based on a search phrase with sorting and pagination"""
    logger.info(f"Fetching news for query: {query}")
    try:
        # GNews parameters
        params = {
            "q": query,
            "token": GNEWS_API_KEY,
            "max": page_size,  # GNews uses 'max' instead of 'pageSize'
            "sortby": sort_by,  # GNews supports: relevance, publishedAt
            "lang": "en"  # Add language filter
        }

        response = requests.get(GNEWS_API_URL, params=params)

        logger.info(f"GNews API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"GNews API error: {response.text}")
            error_message = f"Failed to fetch news. Status code: {response.status_code}"
            try:
                error_detail = response.json()
                error_message += f", Response: {error_detail}"
            except:
                error_message += f", Response text: {response.text}"

            status_code = response.status_code if response.status_code != 0 else 500
            raise HTTPException(status_code=status_code, detail=error_message)

        # GNews response structure is different
        articles = response.json().get("articles", [])

        # Format articles to match your frontend expectations
        valid_articles = [{
            "title": article["title"],
            "url": article["url"],
            "source": article["source"]["name"]
        } for article in articles if validate_url(article["url"])]

        return valid_articles[:page_size]

    except Exception as e:
        logger.error(f"Error in fetch_news: {str(e)}")
        raise

headers = {"Authorization": "Bearer " + HF_TOKEN}

@lru_cache(maxsize=100)
def query(payload_str: str):
    payload = eval(payload_str)  # Convert string to dict
    response = requests.post(INFERENCE_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Error from Hugging Face API")
    return response.json()

@app.get("/analyze/")
async def analyze_article(url: str):
    """Extract, summarize, and analyze sentiment of an article"""
    try:
        # Extract article text
        article = Article(url)
        try:
            article.download()
            article.parse()
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Failed to fetch article: {str(e)}")

        text = article.text
        words = text.split()
        text = ' '.join(words[:MAX_WORDS])

        # Summarize article
        summary = query(str({
            "inputs": text,
        }))[0]["summary_text"]

        # Perform sentiment analysis
        sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]
        sentiment_label = "Positive" if sentiment_score > SENTIMENT_THRESHOLD else "Negative" if sentiment_score < -SENTIMENT_THRESHOLD else "Neutral"

        return {
            "title": article.title,
            "summary": summary,
            "sentiment": sentiment_label
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"API service unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")