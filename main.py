import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from newspaper import Article
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer

# URL validation function
def validate_url(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Load API key
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

app = FastAPI()
port = int(os.environ.get("PORT", 8000))

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NLP models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
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

@app.get("/analyze/")
def analyze_article(url: str):
    """Extract, summarize, and analyze sentiment of an article"""
    try:
        # Extract article text
        article = Article(url)
        article.download()
        article.parse()
        text = article.text

        # Summarize article (max tokens ~130 words)
        summary = summarizer(text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

        # Perform sentiment analysis
        sentiment_score = sentiment_analyzer.polarity_scores(text)["compound"]
        sentiment_label = "Positive" if sentiment_score > 0.05 else "Negative" if sentiment_score < -0.05 else "Neutral"

        return {
            "title": article.title,
            "summary": summary,
            "sentiment": sentiment_label
        }
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid URL or unable to fetch article")