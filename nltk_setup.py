import nltk
import os

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Set NLTK data path so Streamlit Cloud knows where to look
nltk.data.path.append(nltk_data_dir)

# Ensure both punkt and punkt_tab are downloaded
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)
