import os

# Target tickers for daily sentiment tracking
TICKER_LIST = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS", "TATAMOTORS.NS"]

# Threshold for auto-accepting predictions without human review
CONFIDENCE_THRESHOLD = 0.65

# Weight multipliers for news sources reflecting their historical credibility
CREDIBILITY_WEIGHTS = {
    "moneycontrol": 0.95,
    "economic_times": 0.90,
    "livemint": 0.85,
    "pr_newswire": 0.40
}

# Pipeline log paths (using relative paths)
LOG_FILE_PATH = os.path.join("data", "sentiment_log.csv")
MODEL_PATH = "aryanchauhan08/Indi-FinBERT-Model"
