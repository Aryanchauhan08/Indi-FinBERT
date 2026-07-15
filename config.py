import os

# Target tickers for daily sentiment tracking (Expanded to Nifty 50 constituents)
TICKER_LIST = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "LTIM.NS", "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS"
]

# Threshold for auto-accepting predictions without human review
CONFIDENCE_THRESHOLD = 0.65

# Weight multipliers for news sources reflecting their historical credibility
CREDIBILITY_WEIGHTS = {
    "moneycontrol": 0.95,
    "economic_times": 0.90,
    "livemint": 0.85,
    "pr_newswire": 0.40
}

# Ticker keywords mapping for filtering and aligning general scrapers (e.g., MoneyControl)
TICKER_KEYWORDS = {
    "ADANIENT.NS": ["adani enterprises", "adanient"],
    "ADANIPORTS.NS": ["adani ports", "adaniports"],
    "APOLLOHOSP.NS": ["apollo hospitals", "apollohosp"],
    "ASIANPAINT.NS": ["asian paints", "asianpaint"],
    "AXISBANK.NS": ["axis bank", "axisbank"],
    "BAJAJ-AUTO.NS": ["bajaj auto"],
    "BAJFINANCE.NS": ["bajaj finance", "bajfinance"],
    "BAJAJFINSV.NS": ["bajaj finserv", "bajajfinsv"],
    "BPCL.NS": ["bharat petroleum", "bpcl"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel"],
    "BRITANNIA.NS": ["britannia"],
    "CIPLA.NS": ["cipla"],
    "COALINDIA.NS": ["coal india", "coalindia"],
    "DIVISLAB.NS": ["divis laboratories", "divislab"],
    "DRREDDY.NS": ["dr reddy", "drreddy"],
    "EICHERMOT.NS": ["eicher motors", "eichermot"],
    "GRASIM.NS": ["grasim"],
    "HCLTECH.NS": ["hcl technologies", "hcltech"],
    "HDFCBANK.NS": ["hdfc bank", "hdfcbank"],
    "HDFCLIFE.NS": ["hdfc life", "hdfclife"],
    "HEROMOTOCO.NS": ["hero motocorp", "heromotoco"],
    "HINDALCO.NS": ["hindalco"],
    "HINDUNILVR.NS": ["hindustan unilever", "hindunilvr"],
    "ICICIBANK.NS": ["icici bank", "icicibank"],
    "ITC.NS": ["itc"],
    "INDUSINDBK.NS": ["indusind bank", "indusindbk"],
    "INFY.NS": ["infosys", "infy"],
    "JSWSTEEL.NS": ["jsw steel", "jswsteel"],
    "KOTAKBANK.NS": ["kotak mahindra", "kotakbank"],
    "LT.NS": ["larsen & toubro", "lt"],
    "LTIM.NS": ["ltimindtree", "ltim"],
    "M&M.NS": ["mahindra & mahindra", "m&m"],
    "MARUTI.NS": ["maruti suzuki", "maruti"],
    "NESTLEIND.NS": ["nestle india", "nestleind"],
    "NTPC.NS": ["ntpc"],
    "ONGC.NS": ["ongc"],
    "POWERGRID.NS": ["power grid", "powergrid"],
    "RELIANCE.NS": ["reliance industries", "reliance"],
    "SBILIFE.NS": ["sbi life", "sbilife"],
    "SBIN.NS": ["state bank of india", "sbi", "sbin"],
    "SUNPHARMA.NS": ["sun pharma", "sunpharma"],
    "TATAMOTORS.NS": ["tata motors", "tatamotors"],
    "TATASTEEL.NS": ["tata steel", "tatasteel"],
    "TATACONSUM.NS": ["tata consumer", "tataconsum"],
    "TCS.NS": ["tata consultancy services", "tcs"],
    "TECHM.NS": ["tech mahindra", "techm"],
    "TITAN.NS": ["titan company", "titan"],
    "ULTRACEMCO.NS": ["ultratech cement", "ultracemco"],
    "UPL.NS": ["upl"],
    "WIPRO.NS": ["wipro"]
}

# Queries map for Google News RSS searches
TICKER_QUERIES = {t: [t, TICKER_KEYWORDS[t][0].title()] for t in TICKER_LIST}

# Pipeline log paths (using relative paths)
LOG_FILE_PATH = os.path.join("data", "sentiment_log.csv")
MODEL_PATH = "aryanchauhan08/Indi-FinBERT"
