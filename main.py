import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from functools import lru_cache

# Load API key
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")
FRONTEND_FULL_URL = os.getenv("FRONTEND_FULL_URL")
LOCAL_FRONTEND_URL = os.getenv("LOCAL_FRONTEND_URL")
INFERENCE_API_URL = os.getenv("INFERENCE_API_URL")
NEWS_API_URL = os.getenv("NEWS_API_URL")
MAX_WORDS = int(os.getenv("MAX_WORDS"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD"))

app = FastAPI()

# CORS middleware
origins = [
    LOCAL_FRONTEND_URL,
    FRONTEND_URL,
    FRONTEND_FULL_URL,
    "*"
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

@app.get("/fetch_news/")
def fetch_news(query: str, sort_by: str = "relevancy", page_size: int = 10):
    """Fetch news articles based on a search phrase with sorting and pagination"""
    # Start by fetching a larger batch (e.g., 50 articles)
    extra_page_size = 50
    url = f"{NEWS_API_URL}?q={query}&apiKey={NEWS_API_KEY}&sortBy={sort_by}&pageSize={extra_page_size}"
    response = requests.get(url)

    if response.status_code != 200:
      error_message = f"Failed to fetch news. Status code: {response.status_code}"
      try:
        error_detail = response.json()
        error_message += f", Response: {error_detail}"
      except:
        error_message += f", Response text: {response.text}"

      # Use appropriate status code from the News API response
      status_code = response.status_code if response.status_code != 0 else 500
      raise HTTPException(status_code=status_code, detail=error_message)

    articles = response.json().get("articles", [])

    # Validate URLs and filter out invalid ones
    valid_articles = [{"title": a["title"], "url": a["url"], "source": a["source"]["name"]} for a in articles if validate_url(a["url"])]

    # If not enough valid articles, fetch more (if possible)
    while len(valid_articles) < page_size and len(articles) < extra_page_size:
        # Fetch more if needed, adjusting the query or page number
        response = requests.get(url)  # You can modify URL here to fetch the next batch
        if response.status_code != 200:
            break
        articles = response.json().get("articles", [])
        valid_articles.extend([{"title": a["title"], "url": a["url"], "source": a["source"]["name"]} for a in articles if validate_url(a["url"])])

    # Return only up to page_size number of articles
    return valid_articles[:page_size]

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


@app.get("/health")
async def health_check():
    return {"status": "ok"}