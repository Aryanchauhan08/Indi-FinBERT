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
    "M&M.NS": ["mahindra & mahindra", "mahindra suv", "mahindra auto", "m&m auto"],
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
TICKER_QUERIES = {
    "ADANIENT.NS":   ["Adani Enterprises NSE India", "Adani Enterprises stock earnings"],
    "ADANIPORTS.NS": ["Adani Ports India NSE", "APSEZ Adani Ports earnings India"],
    "APOLLOHOSP.NS": ["Apollo Hospitals India NSE", "Apollo Hospitals earnings India"],
    "ASIANPAINT.NS": ["Asian Paints India NSE", "Asian Paints earnings India"],
    "AXISBANK.NS":   ["Axis Bank India NSE", "Axis Bank earnings results India"],
    "BAJAJ-AUTO.NS": ["Bajaj Auto India NSE", "Bajaj Auto sales earnings India"],
    "BAJFINANCE.NS": ["Bajaj Finance India NSE", "Bajaj Finance earnings India"],
    "BAJAJFINSV.NS": ["Bajaj Finserv India NSE", "Bajaj Finserv earnings India"],
    "BPCL.NS":       ["Bharat Petroleum BPCL India NSE", "BPCL earnings India"],
    "BHARTIARTL.NS": ["Bharti Airtel India NSE", "Airtel India earnings results"],
    "BRITANNIA.NS":  ["Britannia Industries India NSE", "Britannia earnings India"],
    "CIPLA.NS":      ["Cipla pharma India NSE", "Cipla earnings results India"],
    "COALINDIA.NS":  ["Coal India NSE", "Coal India earnings production"],
    "DIVISLAB.NS":   ["Divi's Laboratories India NSE", "Divislab earnings India"],
    "DRREDDY.NS":    ["Dr Reddy's Laboratories India NSE", "Dr Reddy earnings India"],
    "EICHERMOT.NS":  ["Eicher Motors India NSE", "Royal Enfield Eicher earnings India"],
    "GRASIM.NS":     ["Grasim Industries India NSE", "Grasim earnings India"],
    "HCLTECH.NS":    ["HCL Technologies India NSE", "HCLTech earnings results India"],
    "HDFCBANK.NS":   ["HDFC Bank India NSE", "HDFC Bank earnings results India"],
    "HDFCLIFE.NS":   ["HDFC Life Insurance India NSE", "HDFC Life earnings India"],
    "HEROMOTOCO.NS": ["Hero MotoCorp India NSE", "Hero MotoCorp sales earnings India"],
    "HINDALCO.NS":   ["Hindalco Industries India NSE", "Hindalco earnings India"],
    "HINDUNILVR.NS": ["Hindustan Unilever India NSE", "HUL earnings results India"],
    "ICICIBANK.NS":  ["ICICI Bank India NSE", "ICICI Bank earnings results India"],
    "ITC.NS":        ["ITC Limited India NSE", "ITC earnings cigarettes FMCG India"],
    "INDUSINDBK.NS": ["IndusInd Bank India NSE", "IndusInd Bank earnings India"],
    "INFY.NS":       ["Infosys India NSE", "Infosys earnings guidance India"],
    "JSWSTEEL.NS":   ["JSW Steel India NSE", "JSW Steel earnings India"],
    "KOTAKBANK.NS":  ["Kotak Mahindra Bank India NSE", "Kotak Bank earnings India"],
    "LT.NS":         ["Larsen Toubro India NSE", "L&T infrastructure earnings India"],
    "LTIM.NS":       ["LTIMindtree India NSE", "LTIMindtree earnings results India"],
    "M&M.NS":        ["Mahindra AND Mahindra automobile India NSE", "M&M auto passenger vehicle India earnings"],
    "MARUTI.NS":     ["Maruti Suzuki India NSE", "Maruti Suzuki sales earnings India"],
    "NESTLEIND.NS":  ["Nestle India NSE", "Nestle India earnings results"],
    "NTPC.NS":       ["NTPC Limited India NSE", "NTPC power earnings India"],
    "ONGC.NS":       ["ONGC India NSE", "Oil Natural Gas Corporation earnings India"],
    "POWERGRID.NS":  ["Power Grid Corporation India NSE", "POWERGRID India transmission earnings"],
    "RELIANCE.NS":   ["Reliance Industries India NSE", "Reliance Industries earnings Jio retail"],
    "SBILIFE.NS":    ["SBI Life Insurance India NSE", "SBI Life earnings India"],
    "SBIN.NS":       ["State Bank India SBI NSE", "SBI bank earnings results India"],
    "SUNPHARMA.NS":  ["Sun Pharma India NSE", "Sun Pharmaceutical earnings India"],
    "TATAMOTORS.NS": ["Tata Motors India NSE", "Tata Motors EV earnings India"],
    "TATASTEEL.NS":  ["Tata Steel India NSE", "Tata Steel earnings production India"],
    "TATACONSUM.NS": ["Tata Consumer Products India NSE", "Tata Consumer earnings India"],
    "TCS.NS":        ["Tata Consultancy Services India NSE", "TCS earnings results guidance India"],
    "TECHM.NS":      ["Tech Mahindra India NSE", "Tech Mahindra earnings results India"],
    "TITAN.NS":      ["Titan Company India NSE", "Titan jewellery earnings India"],
    "ULTRACEMCO.NS": ["UltraTech Cement India NSE", "UltraTech earnings results India"],
    "UPL.NS":        ["UPL Limited agrochemicals India NSE", "UPL Limited India earnings"],
    "WIPRO.NS":      ["Wipro India NSE", "Wipro IT earnings results India"],
}

# Pipeline log paths (using relative paths)
LOG_FILE_PATH = os.path.join("data", "sentiment_log.csv")
MODEL_PATH = "aryanchauhan08/Indi-FinBERT"
