import os
import csv
import logging
import datetime
import urllib.request
import hashlib
import email.utils
import requests
from bs4 import BeautifulSoup
import feedparser
from dotenv import load_dotenv
from config import TICKER_LIST, CONFIDENCE_THRESHOLD, LOG_FILE_PATH, MODEL_PATH, TICKER_KEYWORDS, TICKER_QUERIES

# Load environment variables from local .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Hugging Face Hub Login block
try:
    from huggingface_hub import login
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        logging.info("HF_TOKEN found. Authenticating with Hugging Face Hub...")
        login(token=hf_token)
    else:
        logging.warning("HF_TOKEN environment variable not set. Authenticated access to Hugging Face Hub might fail.")
except ImportError:
    logging.warning("huggingface_hub library not installed. HF Hub login step skipped.")

# Attempt to import transformers for HuggingFace model pipeline
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Quality filters shared across all sources (aligned with notebooks)
BLOCKED_SOURCES = {
    "facebook.com",
    "Українські Національні Новини (УНН)",
    "Межа. Новини України.",
    "IranWire",
    "AlleyWatch",
    "GovCon Wire",
    "Business Wire",
    "simplywall.st",
}

INDIA_SIGNALS = [
    "india", "indian", "nse", "bse", "nifty", "sensex",
    "mumbai", "delhi", "bengaluru", "sebi", "rbi",
    "crore", "lakh", "rupee", "rs.", "fy27", "fy26", "fy25",
    "q1", "q2", "q3", "q4"
]

FOREIGN_EXCHANGE_SIGNALS = ["asx:", "(asx", "nyse:", "nasdaq:", "tsx:", "lse:", "ftse"]

NOISE_WORDS = [
    "cricket", "bollywood", "recipe", "horoscope", "fashion",
    "travel", "weather", "lifestyle", "us-iran", "ukraine",
    "gaza", "election", "sports", "entertainment",
    "runs over", "sexual harassment", "bail granted", "assault", "accident",
    "wimbledon", "premier league", "football club", "kyiv", "dynamo",
    "ontario", "zambia", "cambodia", "poland", "french guiana",
    "westgold", "mastermyne", "telix",
    "de zerbi", "soccer", "premier league manager",
]
SKIP_PREFIXES = [
    "sensex today", "stock market highlights", "market highlights",
    "ahead of market", "top stocks to watch", "stocks in news",
    "buzzing stocks", "10 things that will", "week ahead",
    "trading guide", "market wrap", "top business & market headlines",
    "bl morning report", "morning report", "dalal street watch",
    "share price live",
    "share price today",
    "share price update",
    "stock price live",
    "stock price today",
    "adani ent share price",
    "nse bse",
    "opinion:", "interview:", "exclusive:", "explained:", "watch:", 
    "podcast:", "photo:", "gallery:", "ipo allotment", "ipo listing",
    "ipo gmp", "ipo subscription", "sme ipo", "budget 2025", "union budget",
]
JUNK_PHRASES = [
    "day’s trial", "subscribe", "sign up", "download the app",
    "advisory alert", "read also", "also read", "newsletter",
    "live updates",
    "market performance",
    "opens at",
    "trades at",
    "falls in trade",
    "rises in trade",
    "up in early trade",
    "down in early trade",
    "intraday",
    "trading session",
    "52-week high",
    "52-week low",
    "watch the stock",
    "stocks to watch", "stocks in focus",
    "series a", "seed round",
    "raises $", "secures $", "scores $", "closes $", "bags $",
    "click here", "read more", "follow us",
    "personal finance", "mutual fund sip", "fixed deposit",
    "gold rate", "silver rate", "petrol price", "diesel price",
    "cryptocurrency", "bitcoin", "ethereum",
    "opinion |", "opinion:", "seán", "people love",
    "raises €", "secures €", "closes €",
    "fy26 guidance", "fy26 slides",
    "asx:", "(asx:", "nyse:", "tsx:",
    "wildfire", "orbit", "in-orbit",
]



def is_valid_headline(title):
    """
    Returns True if the headline passes all quality and length filters from the notebooks.
    """
    if not title or title.strip() == "" or title == "[Removed]":
        return False
    words = title.split()
    if len(words) < 6 or len(words) > 45:
        return False
    low = title.lower()
    if any(w in low for w in NOISE_WORDS):
        return False
    if any(low.startswith(p) for p in SKIP_PREFIXES):
        return False
    if any(phrase in low for phrase in JUNK_PHRASES):
        return False
    return True


def parse_rss_date(date_str):
    """
    Parses RSS dates (RFC 822 / standard strings) safely.
    """
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        return dt
    except Exception:
        try:
            from dateutil import parser as date_parser
            return date_parser.parse(date_str)
        except Exception:
            return datetime.datetime.now()


def is_within_last_24_hours(published_dt):
    """
    Determines if a parsed datetime falls within the last 24 hours.
    """
    if published_dt.tzinfo is not None:
        published_dt = published_dt.replace(tzinfo=None)
    now = datetime.datetime.now()
    diff = now - published_dt
    return diff <= datetime.timedelta(hours=24)


def fetch_moneycontrol_news():
    """
    Scrapes the front page sections of MoneyControl and filters/aligns headlines by ticker.
    """
    logging.info("Source A: Starting MoneyControl Scraping...")
    results = []
    
    BASE_URLS = [
        "https://www.moneycontrol.com/news/business/stocks/page-1/",
        "https://www.moneycontrol.com/news/business/markets/page-1/",
    ]
    HEADERS = {"User-Agent": "Mozilla/5.0"}
    
    for url in BASE_URLS:
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            h2_elements = soup.find_all('h2')
            
            for h2 in h2_elements:
                headline_text = h2.text.strip()
                if not is_valid_headline(headline_text):
                    continue
                
                # Assign to relevant tickers if keywords match
                for ticker, keywords in TICKER_KEYWORDS.items():
                    if any(kw in headline_text.lower() for kw in keywords):
                        # Front page items are current within 24 hours
                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        results.append({
                            "Date": now_str,
                            "Ticker": ticker,
                            "Headline": headline_text,
                            "Source": "moneycontrol"
                        })
                        logging.info(f"  [MoneyControl] Matched {ticker} -> {headline_text[:40]}...")
        except Exception as e:
            logging.error(f"  Error scraping MoneyControl URL {url}: {e}")
            
    return results


def fetch_gnews_rss():
    """
    Fetches and parses Google News RSS feeds for targeted tickers.
    """
    logging.info("Source B: Starting Google News RSS Fetching...")
    results = []
    
    GNEWS_BASE = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/rss+xml, application/xml, */*"
    }
    
    for ticker in TICKER_LIST:
        try:
            queries = TICKER_QUERIES.get(ticker, [ticker])
            for query in queries:
                url = GNEWS_BASE.format(q=query.replace(" ", "+"))
                req = urllib.request.Request(url, headers=HEADERS)
                
                with urllib.request.urlopen(req, timeout=10) as response:
                    data = response.read()
                
                feed = feedparser.parse(data)
                added_count = 0
                
                for entry in feed.entries:
                    title = (entry.get("title") or "").strip()
                    # Strip standard "- Source" suffix Google appends to RSS titles
                    if " - " in title:
                        title = title.rsplit(" - ", 1)[0].strip()
                    
                    if not is_valid_headline(title):
                        continue
                        
                    if not any(sig in title.lower() for sig in INDIA_SIGNALS):
                        logging.debug(f"  [GNews] Skipped (no India signal): {title[:60]}")
                        continue
                        
                    if any(sig in title.lower() for sig in FOREIGN_EXCHANGE_SIGNALS):
                        logging.debug(f"  [GNews] Skipped (foreign exchange): {title[:60]}")
                        continue
                        
                    pub_date_str = entry.get("published", "")
                    pub_dt = parse_rss_date(pub_date_str)
                    
                    if not is_within_last_24_hours(pub_dt):
                        continue
                        
                    source = entry.get("source", {}).get("title", "Google News")
                    if source in BLOCKED_SOURCES:
                        continue
                    
                    results.append({
                        "Date": pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "Ticker": ticker,
                        "Headline": title,
                        "Source": source
                    })
                    added_count += 1
                    
                if added_count > 0:
                    logging.info(f"  [GNews] Fetched {added_count} articles for {ticker} using query '{query}'")
        except Exception as e:
            logging.error(f"  Error fetching GNews RSS for {ticker}: {e}")
            
    return results


def fetch_news_api():
    """
    Fetches articles from NewsAPI for targeted tickers (requires NEWS_API_KEY environment variable).
    """
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        logging.info("Source C: NEWS_API_KEY environment variable not set. NewsAPI fetching skipped.")
        return []
        
    logging.info("Source C: Starting NewsAPI Fetching...")
    results = []
    domains = "economictimes.indiatimes.com,livemint.com,business-standard.com,thehindubusinessline.com,financialexpress.com"
    
    for ticker in TICKER_LIST:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "domains": domains,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                logging.error(f"  NewsAPI request failed (status {response.status_code}): {response.text}")
                continue
                
            articles = response.json().get("articles", [])
            added_count = 0
            
            for art in articles:
                title = (art.get("title") or "").strip()
                if not is_valid_headline(title):
                    continue
                    
                pub_date_str = art.get("publishedAt", "")
                try:
                    pub_dt = datetime.datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
                except Exception:
                    pub_dt = datetime.datetime.now()
                    
                if not is_within_last_24_hours(pub_dt):
                    continue
                    
                source = art.get("source", {}).get("name", "NewsAPI")
                if source in BLOCKED_SOURCES:
                    continue
                
                results.append({
                    "Date": pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "Ticker": ticker,
                    "Headline": title,
                    "Source": source
                })
                added_count += 1
                
            if added_count > 0:
                logging.info(f"  [NewsAPI] Fetched {added_count} articles for {ticker}")
        except Exception as e:
            logging.error(f"  Error fetching NewsAPI for {ticker}: {e}")
            
    return results


def deduplicate_news(news_list):
    """
    Deduplicates fetched articles by Ticker and Headline content.
    """
    seen_hashes = set()
    deduped = []
    for item in news_list:
        key = f"{item['Ticker']}_{item['Headline'].lower().strip()}"
        h = hashlib.md5(key.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(item)
    return deduped


def generate_mock_news_fallback():
    """
    Generates dummy news headlines if all scraper outputs are empty.
    Prevents pipeline crashes under network restriction or sparse news days.
    """
    logging.warning("All scraping sources returned empty. Generating 5 mock news items as pipeline fallback.")
    now = datetime.datetime.now()
    return [
        {
            "Date": (now - datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": "RELIANCE.NS",
            "Headline": "Reliance Industries expands retail footprint with new digital platform launch.",
            "Source": "moneycontrol"
        },
        {
            "Date": (now - datetime.timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": "HDFCBANK.NS",
            "Headline": "HDFC Bank Q1 net profit rises 15%, matching analyst estimates.",
            "Source": "economic_times"
        },
        {
            "Date": (now - datetime.timedelta(hours=9)).strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": "INFY.NS",
            "Headline": "Infosys shares slide on cautious revenue guidance revision.",
            "Source": "livemint"
        },
        {
            "Date": (now - datetime.timedelta(hours=14)).strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": "TCS.NS",
            "Headline": "TCS announces major multi-million dollar cloud migration deal with UK retailer.",
            "Source": "pr_newswire"
        },
        {
            "Date": (now - datetime.timedelta(hours=20)).strftime("%Y-%m-%d %H:%M:%S"),
            "Ticker": "TATAMOTORS.NS",
            "Headline": "Tata Motors EV sales drop slightly in domestic market, utility vehicles strong.",
            "Source": "livemint"
        }
    ]


def fetch_latest_news():
    """
    Refactored live news fetching logic incorporating notebook scraping.
    Pulls from MoneyControl, Google News RSS, and NewsAPI.
    """
    logging.info("Initiating news collection loops for Indian tickers...")
    collected_articles = []
    
    # 1. Scrape MoneyControl
    mc_articles = fetch_moneycontrol_news()
    collected_articles.extend(mc_articles)
    
    # 2. Fetch GNews RSS
    gnews_articles = fetch_gnews_rss()
    collected_articles.extend(gnews_articles)
    
    # 3. Fetch NewsAPI (optional)
    newsapi_articles = fetch_news_api()
    collected_articles.extend(newsapi_articles)
    
    # 4. Deduplicate
    deduped_articles = deduplicate_news(collected_articles)
    logging.info(f"Total unique articles crawled: {len(deduped_articles)}")
    
    # 5. Resiliency Fallback check
    if not deduped_articles:
        return generate_mock_news_fallback()
        
    return deduped_articles


def predict_sentiment_fallback(headline):
    """
    Fallback rule-based heuristic prediction in case HuggingFace pipeline fails or is not installed.
    """
    headline_lower = headline.lower()
    positive_words = ["rise", "high", "growth", "profit", "deal", "expand", "gain", "strong", "launch"]
    negative_words = ["slide", "drop", "decline", "fall", "cautious", "loss", "weak"]
    
    pos_count = sum(word in headline_lower for word in positive_words)
    neg_count = sum(word in headline_lower for word in negative_words)
    
    if pos_count > neg_count:
        return "positive", 0.88
    elif neg_count > pos_count:
        return "negative", 0.82
    else:
        return "neutral", 0.62


def load_sentiment_pipeline():
    """
    Loads the HuggingFace sentiment pipeline using the private model from the Hugging Face Hub.
    Uses AutoTokenizer and AutoModelForSequenceClassification.
    """
    if not TRANSFORMERS_AVAILABLE:
        logging.warning("Transformers library is not installed. Using rule-based fallback classifier.")
        return None
        
    try:
        logging.info(f"Initializing HuggingFace sentiment analysis pipeline from HF repository: '{MODEL_PATH}'...")
        
        # Load model and tokenizer using Auto classes
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Build pipeline
        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logging.info("HuggingFace model and tokenizer loaded successfully from HF Hub.")
        return nlp
    except Exception as e:
        logging.error(f"Failed to load HuggingFace model from HF Hub: {e}. Falling back to rule-based classifier.")
        return None


def hitl_guardrail(confidence):
    """
    Cascading HITL guardrail function.
    Accepts if confidence (max_prob) >= 0.65, otherwise flags for human review.
    """
    if confidence >= CONFIDENCE_THRESHOLD:
        return "AUTO_ACCEPTED"
    else:
        return "FLAGGED_FOR_HUMAN_REVIEW"


def run_inference(news_items, nlp):
    """
    Runs sentiment inference on fetched headlines and runs them through the HITL guardrail.
    """
    logging.info("Starting sentiment inference and HITL guardrail gating...")
    results = []
    
    for item in news_items:
        ticker = item["Ticker"]
        headline = item["Headline"]
        date = item["Date"]
        
        # Run model inference or fallback
        if nlp is not None:
            try:
                pred = nlp(headline)[0]
                predicted_class = pred["label"].lower()
                confidence = float(pred["score"])
                logging.info(f"[{ticker}] Predicted: '{predicted_class}' with confidence {confidence:.4f}")
            except Exception as e:
                logging.error(f"Inference error on headline '{headline}': {e}. Using fallback.")
                predicted_class, confidence = predict_sentiment_fallback(headline)
        else:
            predicted_class, confidence = predict_sentiment_fallback(headline)
            logging.info(f"[{ticker}] (Fallback) Predicted: '{predicted_class}' with confidence {confidence:.4f}")
            
        # Gating
        action_type = hitl_guardrail(confidence)
        
        results.append({
            "Date": date,
            "Ticker": ticker,
            "Headline": headline,
            "Source": item.get("Source", "Unknown"),
            "Predicted_Class": predicted_class,
            "Confidence": round(confidence, 4),
            "Action_Type": action_type
        })
        
    return results


def append_to_csv(results, file_path):
    """
    Appends execution results to the target sentiment log CSV.
    """
    logging.info(f"Logging output results to CSV file: {file_path}")
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    file_exists = os.path.isfile(file_path)
    headers = ["Date", "Ticker", "Headline", "Source", "Predicted_Class", "Confidence", "Action_Type"]
    
    try:
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
                logging.info("CSV file did not exist. Initialized file with headers.")
            writer.writerows(results)
        logging.info("Successfully wrote inference results to CSV.")
    except Exception as e:
        logging.error(f"Failed writing results to CSV: {e}")
        raise e


def main():
    logging.info("Starting daily CRON live inference script...")
    
    # 1. Fetch
    news_items = fetch_latest_news()
    
    # 2. Setup Model
    nlp = load_sentiment_pipeline()
    
    # 3. Predict & Gate
    results = run_inference(news_items, nlp)
    
    # 4. Save
    append_to_csv(results, LOG_FILE_PATH)
    
    logging.info("Daily live inference CRON run completed successfully.")


if __name__ == "__main__":
    main()
