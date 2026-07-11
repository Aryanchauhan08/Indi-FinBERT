import os
import csv
import logging
import datetime
from config import TICKER_LIST, CONFIDENCE_THRESHOLD, LOG_FILE_PATH, MODEL_PATH

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


def fetch_latest_news():
    """
    Mock the news fetching logic for Indian tickers.
    Fetches the latest 24 hours of news.
    """
    logging.info("Fetching latest 24 hours of news for Indian tickers...")
    now = datetime.datetime.now()
    
    # 5 dummy sample headlines representing various tickers and sources
    mock_news = [
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
    
    logging.info(f"Successfully fetched {len(mock_news)} sample headlines.")
    return mock_news


def predict_sentiment_fallback(headline):
    """
    Fallback rule-based heuristic prediction in case HuggingFace pipeline fails or is not installed.
    Useful for local off-line development/runs.
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
    Falls back gracefully to None if transformers isn't available.
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
                # pipeline returns e.g. [{'label': 'positive', 'score': 0.95}]
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
            "Predicted_Class": predicted_class,
            "Confidence": round(confidence, 4),
            "Action_Type": action_type
        })
        
    return results


def append_to_csv(results, file_path):
    """
    Appends execution results to the target sentiment log CSV.
    Creates directory and file if they do not exist.
    """
    logging.info(f"Logging output results to CSV file: {file_path}")
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    file_exists = os.path.isfile(file_path)
    headers = ["Date", "Ticker", "Headline", "Predicted_Class", "Confidence", "Action_Type"]
    
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
