import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from newspaper import Article
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# download data for Sentiment Analysis
nltk.download('vader_lexicon')

# URL validation function
def validate_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Load API key
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

MAX_WORDS = int(os.getenv("MAX_WORDS", "130"))
SENTIMENT_THRESHOLD = float(os.getenv("SENTIMENT_THRESHOLD", "0.05"))

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP models
sentiment_analyzer = SentimentIntensityAnalyzer()

@app.get("/fetch_news/")
def fetch_news(query: str, sort_by: str = "relevancy", page_size: int = 10):
    """Fetch news articles based on a search phrase with sorting and pagination"""
    # Start by fetching a larger batch (e.g., 50 articles)
    extra_page_size = 50
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}&sortBy={sort_by}&pageSize={extra_page_size}"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch news")

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

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)

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

        # Summarize article (max tokens ~130 words)
        summary = query({
            "inputs": text,
        })[0]["summary_text"]

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