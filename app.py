import streamlit as st
import wikipedia
import requests
from bs4 import BeautifulSoup
from googletrans import Translator
import pandas as pd
import bs4
import difflib 
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk_setup

import yake 
bs4.BeautifulSoup.DEFAULT_PARSER = "lxml"
st.set_page_config(page_title="SmartSearch AI", page_icon="🌍", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>SmartSearch</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center; color: gray;'>An Intelligent Multilingual Search Summarizer and Info Recommender</h3>",
    unsafe_allow_html=True
)

translator = Translator()
HEADERS = {"User-Agent": "SmartSearchAI/1.0 (contact: sairam@example.com)"}

nltk.download('punkt', quiet=True)
@st.cache_data(show_spinner=False)
def summarize_text(text, sentences_count=5):
    """
    Generates a custom extractive summary using LSA from Sumy.
    """
    try:
        main_content = text.split("\n== ")[0]
        
        parser = PlaintextParser.from_string(main_content, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count)
        
        return " ".join([str(sentence) for sentence in summary])
    
    except Exception as e:
        st.warning(f"Custom summarization failed: {e}")
        return " ".join(text.splitlines()[:sentences_count])

@st.cache_data(show_spinner=False)
def get_nlp_keywords(text, num_keywords=9):
    """
    Extracts key concepts from the text using YAKE (NLP).
    """
    try:

        main_content = text.split("\n== ")[0]
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=num_keywords, features=None)
        keywords = kw_extractor.extract_keywords(main_content)
        return [kw[0] for kw in keywords]
    except Exception as e:
        st.warning(f"Keyword extraction failed: {e}")
        return []

@st.cache_data(show_spinner="Searching Wikipedia...")
def get_valid_wiki_page(title):
    """
    Try fetching a valid Wikipedia page, skip disambiguations.
    Returns (data_dict, search_suggestions_list)
    """
    
    search_results = []
    try:
        search_results = wikipedia.search(title)
        if not search_results:
            return None, [] 
    except Exception as e:
        st.error(f"Error during Wikipedia search: {e}")
        return None, [] 
    try:
        for result in search_results:
            try:
                page = wikipedia.page(result, auto_suggest=False)
                if page.content and not page.title.lower().endswith("(disambiguation)"):
                    url = page.url
                    r = requests.get(url, headers=HEADERS, timeout=8)
                    soup = BeautifulSoup(r.text, "html.parser")
                    image_url = None
                    infobox = soup.find("table", {"class": "infobox"})
                    if infobox:
                        img_tag = infobox.find("img")
                        if img_tag and img_tag.get("src"):
                            image_url = "https:" + img_tag["src"]
                    info_data = []
                    if infobox:
                        for row in infobox.find_all("tr"):
                            header = row.find("th")
                            data = row.find("td")
                            if header and data:
                                info_data.append((header.text.strip(), data.text.strip()))
                    return {
                        "title": page.title,
                        "summary": summarize_text(page.content, 5),
                        "content": page.content,
                        "url": url,
                        "image": image_url,
                        "info": info_data,
                    }, search_results[1:6]
            
            except wikipedia.DisambiguationError:
                continue 
            except Exception:
                continue 
        return None, search_results

    except Exception as e:
        st.error(f"Error fetching page content: {e}")
        return None, search_results
@st.cache_data(show_spinner=False)
def translate_text(text, lang="te"):
    try:
        return translator.translate(text, dest=lang).text
    except Exception:
        return "Translation failed."
@st.cache_data(show_spinner=False)
def get_search_suggestions(query):
    """Live suggestions using DuckDuckGo autocomplete endpoint."""
    if not query:
        return []
    try:
        url = f"https://duckduckgo.com/ac/?q={query}"
        res = requests.get(url, headers=HEADERS, timeout=4)
        if res.ok and res.text.strip():
            data = res.json()
            return [item.get("phrase") for item in data if item.get("phrase")]
        return []
    except Exception:
        return []
@st.cache_data(show_spinner=False)
def get_web_results(query):
    """Fallback to DuckDuckGo instant API for web results."""
    try:
        url = f"https://api.duckduckgo.com/?q={query}&format=json&pretty=1"
        r = requests.get(url, headers=HEADERS, timeout=8)
        if not r.ok:
            return []
        ddg = r.json()
        results = []
        for topic in ddg.get("RelatedTopics", []):
            if isinstance(topic, dict) and "Text" in topic and "FirstURL" in topic:
                results.append((topic["Text"], topic["FirstURL"]))
            elif isinstance(topic, dict) and "Name" in topic and "Topics" in topic:
                for sub in topic["Topics"]:
                    if "Text" in sub and "FirstURL" in sub:
                        results.append((sub["Text"], sub["FirstURL"]))
        return results
    except Exception:
        return []

if "query" not in st.session_state:
    st.session_state.query = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "current_index" not in st.session_state:
    st.session_state.current_index = -1
def update_history(q):
    """Adds a new query to the history."""
    if st.session_state.current_index < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.current_index + 1]
    st.session_state.history.append(q)
    st.session_state.current_index += 1

def handle_query_click(new_query):
    """Callback to update query from a button click."""
    st.session_state.query = new_query
    if not st.session_state.history or st.session_state.history[-1] != new_query:
        update_history(new_query)

def go_back():
    """Callback for 'Back' button."""
    if st.session_state.current_index > 0:
        st.session_state.current_index -= 1
        st.session_state.query = st.session_state.history[st.session_state.current_index]

def go_forward():
    """Callback for 'Forward' button."""
    if st.session_state.current_index < len(st.session_state.history) - 1:
        st.session_state.current_index += 1
        st.session_state.query = st.session_state.history[st.session_state.current_index]

query_col, back_col, fwd_col = st.columns([6, 1, 1])
with query_col:
    st.text_input("Enter your search query:", key="query", label_visibility="collapsed", placeholder="Enter your search query...")

with back_col:
    st.button("⬅ Back", on_click=go_back, disabled=st.session_state.current_index <= 0, use_container_width=True)
with fwd_col:
    st.button("Forward", on_click=go_forward, disabled=st.session_state.current_index >= len(st.session_state.history) - 1, use_container_width=True)

if st.session_state.query and len(st.session_state.query.strip()) > 1:
    suggestions = get_search_suggestions(st.session_state.query)
    if suggestions:
        st.markdown("##### Suggestions:")
        cols = st.columns(3)
        for i, s in enumerate(suggestions[:9]):
            with cols[i % 3]:
                st.button(s, key=f"sugg_{s}", on_click=handle_query_click, args=(s,), use_container_width=True)
query = st.session_state.query
if query and len(query.strip()) > 1:
    if not st.session_state.history or query != st.session_state.history[-1]:
        update_history(query)
    data, related_searches = get_valid_wiki_page(query)

    if data:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"## {data['title']}")
            st.markdown(f"[🌐 Open on Wikipedia]({data['url']})")
            st.markdown("### Summary (NLP Generated):")
            st.info(data["summary"])
            langs = {"English": "en", "Telugu": "te", "Hindi": "hi", "Spanish": "es", "French": "fr", "German": "de"}
            lang_choice = st.selectbox("🌐 Translate summary to:", list(langs.keys()), key="lang_select")
            if lang_choice != "English":
                translated = translate_text(data["summary"], langs[lang_choice])
                st.markdown(f"### {lang_choice} Translation:")
                st.success(translated)

            st.markdown("### 📚 More Info & Deep Insights:")
            st.write(data["content"][:1500] + "...")

        with col2:
            if data["image"]:
                st.image(data["image"], width=300, caption=data["title"])
            if data["info"]:
                st.markdown("### 🗂️ Key Info:")
                df = pd.DataFrame(data["info"], columns=["Attribute", "Details"])
                st.table(df.head(8))
        st.divider()
        st.markdown("### 🔗 Related Searches (from Wikipedia):")
        if related_searches:
            cols = st.columns(3)
            for i, result in enumerate(related_searches):
                with cols[i % 3]:
                    st.button(result, key=f"rel_{result}", on_click=handle_query_click, args=(result,), use_container_width=True)
        else:
            st.info("No related searches found.")
        st.divider()
        st.markdown("### 🔑 Key Concepts (NLP Extracted)")
        st.caption("Key phrases automatically extracted from the article content.")
        
        nlp_keywords = get_nlp_keywords(data["content"], 9) 
        if nlp_keywords:
            cols = st.columns(3)
            for i, keyword in enumerate(nlp_keywords):
                with cols[i % 3]:
                    st.button(keyword.title(), 
                              key=f"nlp_{keyword}", 
                              on_click=handle_query_click, 
                              args=(keyword,),
                              use_container_width=True)
        else:
            st.info("No key concepts were extracted.")
        st.divider()
        st.markdown("### 🌍 Web Results from DuckDuckGo")
        web_results = get_web_results(query)
        if web_results:
            for text, link in web_results[:10]:
                st.markdown(f"🔗 [{text}]({link})")
        else:
            st.info("No web results found.")

    else:
        st.warning(f"No direct Wikipedia page found for '{query}'.")
        if related_searches: 
            close_matches = difflib.get_close_matches(query, related_searches, n=3, cutoff=0.4)
            
            if close_matches:
                st.markdown("### Did you mean:")
                cols = st.columns(3)
                for i, suggestion in enumerate(close_matches):
                    with cols[i % 3]:
                        st.button(suggestion, 
                                  key=f"didyoumean_{i}", 
                                  on_click=handle_query_click, 
                                  args=(suggestion,),
                                  use_container_width=True)
        st.divider()
        st.markdown("### 🌐 Showing web results instead...")
        web_results = get_web_results(query)
        if web_results:
            for text, link in web_results[:10]:
                st.markdown(f"🔗 [{text}]({link})")
        else:
            st.info("No web results found. Try a different query.")
else:
    st.info("Type a search query above to begin")