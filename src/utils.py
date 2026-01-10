import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()

def detect_link(text):
    text = str(text) # Ensure string
    link_pattern = r"(http|https|www|\.com|\.ph|\.net|\.org|\.gov|\.ly)"
    return bool(re.search(link_pattern, text, re.IGNORECASE))
