# Tunisie Telecom Q&A Backend API
# This script exposes the Q&A logic as a FastAPI HTTP API

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import logging
import re
import time
import json
import unicodedata
import html
import hashlib
from datetime import datetime
import requests
import pandas as pd
from ollama_client import OllamaClient
from bs4 import BeautifulSoup, Comment
from urllib.parse import quote_plus
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# --- Config ---
class Config:
    CSV_FILE = os.path.join(os.path.dirname(__file__), "File.csv")
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
    CACHE_EXPIRY = 60 * 60 * 24 * 7  # 1 week
    USE_CACHE = True
    ENABLE_WEB_SEARCH = True
    MAX_WEB_RESULTS = 3
    MAX_CHARS_PER_PAGE = 3000
    MAX_ANSWER_LENGTH = 600
    SHORT_ANSWER_SENTENCES = 2
    LOCAL_HIGH_CONFIDENCE = 0.85
    LOCAL_MODERATE_CONFIDENCE = 0.65
    LOCAL_LOW_CONFIDENCE = 0.45
    REQUEST_TIMEOUT = 10
    
    # Ollama configuration
    OLLAMA_ENABLED = True
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3"  # Default model to use
    OLLAMA_TEMPERATURE = 0.7
    OLLAMA_MAX_TOKENS = 1024

    @staticmethod
    def get_paths():
        return {
            'model_path': os.path.join(Config.MODEL_DIR, 'qna_model.joblib'),
            'tfidf_path': os.path.join(Config.MODEL_DIR, 'tfidf_vectorizer.joblib'),
            'cache_path': os.path.join(Config.CACHE_DIR, 'web_content_cache.json')
        }

# --- Ollama Client ---
class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url=Config.OLLAMA_URL, model=Config.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self.logger = logging.getLogger("TT_QA.OllamaClient")
        self.logger.info(f"Ollama client initialized with model: {model}")
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                self.logger.error(f"Failed to get models. Status: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama: {str(e)}")
            return []
    
    def is_available(self):
        """Check if Ollama service is available"""
        try:
            models = self.get_available_models()
            return len(models) > 0
        except:
            return False
            
    def generate(self, prompt, system_prompt=None, temperature=Config.OLLAMA_TEMPERATURE, max_tokens=Config.OLLAMA_MAX_TOKENS):
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
                
            self.logger.debug(f"Sending request to Ollama: {self.api_endpoint}")
            response = requests.post(self.api_endpoint, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama")
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from Ollama (Status: {response.status_code})"
                
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error connecting to Ollama: {str(e)}"

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TT_QA")

# --- NLTK Setup ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('french') + stopwords.words('english'))

# --- TT Keywords ---
TT_KEYWORDS = [
    "tunisie telecom", "tt", "tunisietelecom", "ttnet", "elissa", "tt wifi", "tt box", "tt recharge",
    "tt assistance", "tt mobile", "tt internet", "tt fibre", "tt 4g", "tt 5g", "tt offre", "tt code", "tt ussd"
]
OFF_TOPIC_KEYWORDS = ["pizza", "weather", "football", "movie", "recipe", "restaurant", "game", "music", "song", "car", "hotel", "flight", "train", "bus", "metro", "covid", "corona", "virus", "pandemic", "bitcoin", "crypto", "stock", "bourse", "bank", "loan", "insurance", "tax", "visa", "passport", "embassy", "immigration", "university", "school", "exam", "test", "doctor", "hospital", "pharmacy", "medicine", "drug", "disease", "symptom", "treatment", "health", "weather", "temperature", "rain", "sun", "cloud", "wind", "storm", "earthquake", "volcano", "tsunami", "flood", "fire", "accident", "crime", "police", "law", "court", "judge", "prison", "jail", "army", "military", "war", "peace", "election", "vote", "president", "minister", "government", "parliament", "politics", "party", "campaign", "protest", "strike", "union", "worker", "job", "work", "salary", "pay", "bonus", "holiday", "vacation", "travel", "tourism", "trip", "tour", "guide", "map", "direction", "address", "location", "place", "city", "village", "country", "continent", "ocean", "sea", "lake", "river", "mountain", "desert", "forest", "park", "garden", "zoo", "museum", "library", "cinema", "theater", "concert", "festival", "event", "party", "wedding", "birthday", "anniversary", "funeral", "marriage", "divorce", "baby", "child", "kid", "boy", "girl", "man", "woman", "father", "mother", "parent", "family", "friend", "neighbor", "colleague", "boss", "manager", "director", "owner", "customer", "client", "user", "member", "guest", "visitor", "tourist", "citizen", "resident", "foreigner", "immigrant", "refugee", "student", "teacher", "professor", "doctor", "nurse", "engineer", "architect", "lawyer", "judge", "policeman", "soldier", "driver", "pilot", "captain", "sailor", "fisherman", "farmer", "worker", "employee", "employer", "businessman", "businesswoman", "entrepreneur", "investor", "shareholder", "partner", "owner", "boss", "manager", "director", "president", "minister", "mayor", "governor", "ambassador", "diplomat", "officer", "official", "agent", "representative", "delegate", "envoy", "messenger", "spokesman", "spokeswoman", "journalist", "reporter", "editor", "writer", "author", "poet", "artist", "painter", "sculptor", "musician", "singer", "composer", "conductor", "actor", "actress", "dancer", "athlete", "player", "coach", "trainer", "referee", "umpire", "judge", "lawyer", "attorney", "advocate", "counsel", "solicitor", "barrister", "prosecutor", "defender", "witness", "victim", "suspect", "criminal", "prisoner", "inmate", "convict", "offender", "culprit", "accused", "plaintiff", "defendant", "appellant", "respondent", "petitioner", "complainant", "claimant", "applicant", "beneficiary", "recipient", "heir", "successor", "predecessor", "ancestor", "descendant", "relative", "kin"]

class TextProcessor:
    """Enhanced text processing utilities"""
    
    def __init__(self):
        """Initialize text processing components"""
        self.stopwords = nltk_stopwords
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
    
    def normalize_text(self, text):
        """Normalize text by removing accents and converting to lowercase"""
        if not isinstance(text, str):
            return ""
        # Decode HTML entities
        text = html.unescape(text)
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        # Remove non-ascii characters
        text = ''.join(c for c in text if not unicodedata.combining(c))
        # Convert to lowercase
        text = text.lower()
        return text
    
    def basic_clean(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        text = self.normalize_text(text)
        # Remove non-word characters except spaces and preserve hashtags, asterisks, and question marks
        text = re.sub(r'[^\w\s#*?]', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def advanced_clean(self, text):
        """Advanced text cleaning with lemmatization and stopword removal"""
        if not isinstance(text, str):
            return ""
        text = self.normalize_text(text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                  if token.isalnum() and token not in self.stopwords]
        # Rejoin tokens
        return ' '.join(tokens)

# --- Google Search Web Content Manager ---
class GoogleSearchManager:
    """Manager for Google search and web content processing"""
    
    def __init__(self):
        """Initialize Google search manager"""
        self.cache = {}
        self.cache_expiry = {}
        self.load_cache()
        self.text_processor = TextProcessor()
    
    def load_cache(self):
        """Load cache from file"""
        cache_path = Config.get_paths()['cache_path']
        if Config.USE_CACHE and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.cache = cache_data.get('content', {})
                    self.cache_expiry = cache_data.get('expiry', {})
                logger.info(f"Loaded {len(self.cache)} items from web content cache")
            except Exception as e:
                logger.error(f"Failed to load cache: {str(e)}")
                self.cache = {}
                self.cache_expiry = {}
    
    def save_cache(self):
        """Save cache to file"""
        if not Config.USE_CACHE:
            return
            
        cache_path = Config.get_paths()['cache_path']
        cache_dir = os.path.dirname(cache_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        try:
            cache_data = {
                'content': self.cache,
                'expiry': self.cache_expiry,
                'timestamp': datetime.now().isoformat()
            }
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(self.cache)} items to web content cache")
        except Exception as e:
            logger.error(f"Failed to save cache: {str(e)}")
    
    def _get_cache_key(self, url):
        """Generate a cache key for a URL"""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def extract_main_content(self, html_content, query_terms):
        """Extract relevant content from HTML"""
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove unnecessary elements
        for tag in soup(["script", "style", "header", "footer", "nav", "aside", "form", "button", "figcaption"]):
            tag.decompose()
        
        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        relevant_texts = []
        
        for p in paragraphs:
            p_text = p.get_text(separator=' ', strip=True)
            if len(p_text.split()) < 5:
                continue
                
            # Score paragraph relevance to query
            score = sum(1 for term in query_terms if term.lower() in p_text.lower())
            if score > 0:
                relevant_texts.append(p_text)
        
        # If no relevant paragraphs, get all text from body
        if not relevant_texts and soup.body:
            body_text = soup.body.get_text(separator=' ', strip=True)
            if body_text:
                relevant_texts.append(body_text[:Config.MAX_CHARS_PER_PAGE//2])
        
        return " ".join(relevant_texts)[:Config.MAX_CHARS_PER_PAGE]
    
    def fetch_web_content(self, url, query_terms):
        """Fetch content from a web URL with caching"""
        if not url:
            return None
        
        cache_key = self._get_cache_key(url)
        
        # Check cache first if enabled
        if Config.USE_CACHE:
            now = time.time()
            if cache_key in self.cache and cache_key in self.cache_expiry:
                if now < self.cache_expiry[cache_key]:
                    logger.info(f"Using cached content for: {url}")
                    return self.cache[cache_key]
        
        try:
            logger.info(f"Fetching content from: {url}")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5,fr;q=0.3',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
            }
            
            response = requests.get(url, headers=headers, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            # Check if response is HTML
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
                logger.warning(f"URL {url} returned non-HTML content: {content_type}")
                return None
            
            # Process the HTML content
            extracted_content = self.extract_main_content(response.text, query_terms)
            
            # Cache the content if enabled
            if Config.USE_CACHE and extracted_content:
                self.cache[cache_key] = extracted_content
                self.cache_expiry[cache_key] = time.time() + Config.CACHE_EXPIRY
                # Periodically save cache
                if len(self.cache) % 10 == 0:
                    self.save_cache()
            
            return extracted_content
        
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None
    
    def google_search(self, query, site_filter=None, num_results=10):
        """Perform a Google search and return results"""
        try:
            # Format the query with site filter if provided
            formatted_query = query
            if site_filter:
                formatted_query = f"{query} site:{site_filter}"
            
            # URL encode the query
            encoded_query = quote_plus(formatted_query)
            
            # Construct the search URL
            search_url = f"https://www.google.com/search?q={encoded_query}&num={num_results}&hl=en"
            
            # Send the request with browser-like headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5,fr;q=0.3'
            }
            
            logger.info(f"Searching Google for: {formatted_query}")
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            results = []
            
            # Find result blocks
            search_divs = soup.find_all('div', class_=['g', 'tF2Cxc'])
            
            for div in search_divs:
                # Extract title and link
                title_elem = div.find('h3')
                if not title_elem:
                    continue
                
                title = title_elem.get_text()
                
                # Find URL
                link_elem = div.find('a')
                if not link_elem:
                    continue
                
                href = link_elem.get('href', '')
                if href.startswith('/url?'):
                    href = re.search(r'url\?q=([^&]+)', href).group(1)
                
                if not href.startswith('http'):
                    continue
                
                # Extract snippet
                snippet_elem = div.find('div', class_=['VwiC3b', 'yXK7lf', 'MUxGbd', 'yDYNvb', 'lyLwlc'])
                snippet = snippet_elem.get_text() if snippet_elem else ""
                
                results.append({
                    'title': title,
                    'href': href,
                    'snippet': snippet
                })
            
            logger.info(f"Found {len(results)} Google search results")
            return results[:num_results]
            
        except Exception as e:
            logger.error(f"Google search error: {str(e)}")
            return []
    
    def search_web(self, query):
        """Search the web for information about the query"""
        logger.info(f"Performing web search for: {query}")
        query_terms = self.text_processor.basic_clean(query).split()
        
        # Create search variations to get better coverage
        search_configs = [
            {"query": query, "site": "tunisietelecom.tn", "source_type": "Official Site"},
            {"query": f"Tunisie Telecom {query}", "site": None, "source_type": "General Web"},
            {"query": f"{query} Tunisie Telecom facebook", "site": "facebook.com", "source_type": "Social Media"},
        ]
        
        all_results = []
        processed_urls = set()
        
        for config in search_configs:
            try:
                # Perform Google search
                results = self.google_search(
                    query=config["query"],
                    site_filter=config["site"],
                    num_results=Config.MAX_WEB_RESULTS
                )
                
                for result in results:
                    url = result.get('href')
                    title = result.get('title', '')
                    
                    # Skip already processed URLs
                    if not url or url in processed_urls:
                        continue
                    
                    processed_urls.add(url)
                    
                    # Try to fetch content
                    content = self.fetch_web_content(url, query_terms)
                    if content:
                        all_results.append({
                            'url': url,
                            'title': title,
                            'content': content,
                            'source_type': config["source_type"],
                            'snippet': result.get('snippet', '')
                        })
                    elif result.get('snippet'):
                        # Use snippet as fallback
                        all_results.append({
                            'url': url,
                            'title': title,
                            'content': result['snippet'],
                            'source_type': config["source_type"],
                            'snippet': result['snippet']
                        })
                        
            except Exception as e:
                logger.error(f"Search error for {config['query']}: {str(e)}")
                
                # If search fails, try direct access to common TT URLs (fallback)
                if config["source_type"] == "Official Site":
                    common_tt_urls = [
                        "https://www.tunisietelecom.tn/particulier/assistance/",
                        "https://www.tunisietelecom.tn/particulier/mobile/options-services/",
                        f"https://www.tunisietelecom.tn/particulier/recherche/?q={quote_plus(query)}"
                    ]
                    
                    for direct_url in common_tt_urls:
                        if direct_url in processed_urls:
                            continue
                            
                        processed_urls.add(direct_url)
                        content = self.fetch_web_content(direct_url, query_terms)
                        if content:
                            all_results.append({
                                'url': direct_url,
                                'title': "Tunisie Telecom",
                                'content': content,
                                'source_type': "Official Site (Direct)",
                                'snippet': ""
                            })
        
        # Sort results by source priority and prefer official TT website
        def sort_key(result):
            source_priority = {"Official Site": 0, "Official Site (Direct)": 0, "General Web": 1, "Social Media": 2}
            url_priority = 0 if "tunisietelecom.tn" in result.get('url', '') else 1
            return (source_priority.get(result['source_type'], 99), url_priority)
        
        all_results.sort(key=sort_key)
        
        return all_results[:Config.MAX_WEB_RESULTS]

# --- Enhanced QnA Model ---
class EnhancedQnAModel:
    """Enhanced Question-Answering model"""
    
    def __init__(self, csv_path=Config.CSV_FILE):
        """Initialize the QnA model"""
        self.csv_path = csv_path
        self.paths = Config.get_paths()
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.vectorizer = None
        self.nn_model = None
        
        # Data containers
        self.df = None
        self.original_questions = []
        self.cleaned_questions = []
        self.answers = []
        
        # Load or train the model
        self.load_or_train_model()
    
    def load_or_train_model(self):
        """Load a pre-trained model if available, otherwise train a new one"""
        if self._load_model():
            logger.info("Loaded pre-trained QnA model")
        else:
            logger.info("Training new QnA model")
            self._train_model()
            self._save_model()
    
    def _load_model(self):
        """Load model from files if available"""
        try:
            # Check if all required files exist
            model_path = self.paths['model_path']
            tfidf_path = self.paths['tfidf_path']
            
            if not os.path.exists(model_path) or not os.path.exists(tfidf_path):
                return False
            
            # Load data from CSV first
            self._load_csv_data()
            
            # Load models
            nn_data = joblib.load(model_path)
            self.nn_model = nn_data['nn_model']
            self.original_questions = nn_data['original_questions']
            self.cleaned_questions = nn_data['cleaned_questions']
            self.answers = nn_data['answers']
            
            self.vectorizer = joblib.load(tfidf_path)
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _save_model(self):
        """Save model to files"""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.paths['model_path']), exist_ok=True)
            
            # Save NN model and data
            nn_data = {
                'nn_model': self.nn_model,
                'original_questions': self.original_questions,
                'cleaned_questions': self.cleaned_questions,
                'answers': self.answers
            }
            joblib.dump(nn_data, self.paths['model_path'])
            
            # Save vectorizer
            joblib.dump(self.vectorizer, self.paths['tfidf_path'])
            
            logger.info("Model saved successfully")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def _load_csv_data(self):
        """Load and preprocess data from CSV file"""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                return False
            
            self.df = pd.read_csv(self.csv_path)
            
            if 'question' not in self.df.columns or 'answer' not in self.df.columns:
                logger.error("CSV must have 'question' and 'answer' columns")
                return False
            
            # Remove rows with missing values
            self.df.dropna(subset=['question', 'answer'], inplace=True)
            
            if self.df.empty:
                logger.error("CSV contains no valid data after preprocessing")
                return False
            
            # Clean and preprocess texts
            self.df['cleaned_question'] = self.df['question'].apply(self.text_processor.basic_clean)
            
            # Remove rows with empty questions after cleaning
            self.df = self.df[self.df['cleaned_question'].str.strip() != '']
            
            # Store data in instance variables
            self.original_questions = self.df['question'].tolist()
            self.cleaned_questions = self.df['cleaned_question'].tolist()
            self.answers = self.df['answer'].tolist()
            
            logger.info(f"Loaded {len(self.original_questions)} QA pairs from CSV")
            return True
        
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            return False
    
    def _train_model(self):
        """Train the QnA model using the data from CSV"""
        if not self._load_csv_data():
            logger.error("Failed to load CSV data. Model training aborted.")
            return
        
        try:
            # Train TF-IDF model with enhanced parameters
            logger.info("Training TF-IDF model...")
            tfidf_params = {
                'ngram_range': (1, 2),  # Use unigrams and bigrams
                'min_df': 1,
                'max_df': 0.95,
                'analyzer': 'word',
                'use_idf': True,
                'smooth_idf': True,
                'sublinear_tf': True
            }
            self.vectorizer = TfidfVectorizer(**tfidf_params)
            
            # Fit TF-IDF vectorizer on cleaned questions
            X_tfidf = self.vectorizer.fit_transform(self.cleaned_questions)
            
            # Train nearest neighbors model with optimal settings
            logger.info("Training nearest neighbors model...")
            self.nn_model = NearestNeighbors(
                n_neighbors=min(5, len(self.cleaned_questions)),  # Multiple neighbors for robustness
                metric='cosine',  # Cosine similarity is best for text
                algorithm='brute'  # Most accurate for small to medium datasets
            )
            self.nn_model.fit(X_tfidf)
            
            logger.info("Model training completed successfully")
        
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def get_answer_with_details(self, query):
        """Get answer for a query with detailed matching information"""
        if not isinstance(query, str) or not query.strip():
            return {
                'answer': None,
                'matched_question': None,
                'similarity': 0.0,
                'index': -1,
                'top_matches': []
            }
        
        # Clean the query
        cleaned_query = self.text_processor.basic_clean(query)
        
        # Get TF-IDF matches
        try:
            query_vector = self.vectorizer.transform([cleaned_query])
            distances, indices = self.nn_model.kneighbors(query_vector)
            
            if len(indices) > 0 and len(indices[0]) > 0:
                top_matches = []
                for i, idx in enumerate(indices[0]):
                    sim = 1 - distances[0][i]
                    top_matches.append({
                        'question': self.original_questions[idx],
                        'similarity': float(sim),
                        'index': int(idx)
                    })
                
                # Best match is first in array
                best_idx = indices[0][0]
                best_sim = 1 - distances[0][0]
                
                return {
                    'answer': self.answers[best_idx],
                    'matched_question': self.original_questions[best_idx],
                    'similarity': float(best_sim),
                    'index': int(best_idx),
                    'method': 'tfidf',
                    'top_matches': top_matches[:3]  # Only return top 3 matches
                }
            
        except Exception as e:
            logger.error(f"Error in question matching: {str(e)}")
        
        return {
            'answer': None,
            'matched_question': None,
            'similarity': 0.0,
            'index': -1,
            'top_matches': []
        }

# --- Answer Generation ---
class AnswerGenerator:
    """Generate high-quality answers"""
    
    def __init__(self):
        """Initialize the answer generator"""
        self.text_processor = TextProcessor()
    
    def summarize_text(self, text, num_sentences=Config.SHORT_ANSWER_SENTENCES):
        """Summarize text by extracting most important sentences"""
        if not text:
            return "Aucune information disponible."
        
        # For short texts, just return them directly
        if len(text.split()) < 20:
            return text
            
        # Split into sentences
        sentences = sent_tokenize(text)
        
        # Filter out very short or irrelevant sentences
        meaningful_sentences = []
        for s in sentences:
            if len(s.split()) < 4:
                continue
                
            # Skip generic sentences that don't add value
            skip_patterns = [
                "cookie", "javascript", "terms of use", "privacy policy",
                "all rights reserved", "learn more", "click here", "pour plus d'informations"
            ]
            if any(pattern in s.lower() for pattern in skip_patterns):
                continue
                
            meaningful_sentences.append(s)
        
        if not meaningful_sentences:
            meaningful_sentences = [s for s in sentences if len(s.split()) > 3]
        
        if not meaningful_sentences:
            return "Information non disponible."
        
        # Take top N sentences
        summary = " ".join(meaningful_sentences[:num_sentences]).strip()
        
        # Truncate if too long
        if len(summary) > Config.MAX_ANSWER_LENGTH:
            summary = summary[:Config.MAX_ANSWER_LENGTH] + "..."
        
        return summary
    
    def format_answer_with_source(self, answer_text, source, confidence):
        """Format the answer with source attribution"""
        if source.startswith("LOCAL"):
            prefix = "[TT Knowledge Base]"
            if "LOW_CONF" in source:
                prefix = "[TT Knowledge Base (low confidence)]"
            elif "MOD_CONF" in source:
                prefix = "[TT Knowledge Base (moderate confidence)]"
            elif "HIGH_CONF" in source:
                prefix = "[TT Knowledge Base (high confidence)]"
            
            return f"{prefix}: {self.summarize_text(answer_text)}"
        
        elif source == "WEB_TT_FOUND":
            return f"[Web Info (Tunisie Telecom)]: {self.summarize_text(answer_text)}"
        
        return self.summarize_text(answer_text)
    
    def combine_sources(self, local_result, web_results):
        """Combine local and web sources to produce the best answer"""
        local_similarity = local_result.get('similarity', 0)
        local_answer = local_result.get('answer')
        
        result = {
            'answer': None,
            'source': "UNKNOWN",
            'confidence': 0,
            'informative': False
        }
        
        # 1. High confidence local answer
        if local_similarity >= Config.LOCAL_HIGH_CONFIDENCE and local_answer:
            result['answer'] = local_answer
            result['source'] = "LOCAL_HIGH_CONF"
            result['confidence'] = local_similarity
            result['informative'] = True
        
        # 2. Moderate confidence local answer
        elif local_similarity >= Config.LOCAL_MODERATE_CONFIDENCE and local_answer:
            result['answer'] = local_answer
            result['source'] = "LOCAL_MOD_CONF"
            result['confidence'] = local_similarity
            result['informative'] = True
        
        # 3. Web results available
        elif web_results:
            # Get best web result (already sorted by priority)
            best_web = web_results[0]
            result['answer'] = best_web['content']
            result['source'] = "WEB_TT_FOUND"
            result['confidence'] = 0.6  # Standard confidence for web results
            result['informative'] = True
            result['url'] = best_web.get('url')
        
        # 4. Low confidence local answer as fallback
        elif local_similarity >= Config.LOCAL_LOW_CONFIDENCE and local_answer:
            result['answer'] = local_answer
            result['source'] = "LOCAL_LOW_CONF"
            result['confidence'] = local_similarity
            result['informative'] = True
        
        # 5. No good match found
        else:
            result['answer'] = f"Je n'ai pas d'information spécifique sur ce sujet. Veuillez consulter le site officiel de Tunisie Telecom."
            result['source'] = "TT_NOT_IN_KB_NO_WEB"
            result['confidence'] = 0
            result['informative'] = False
        
        return result

# --- TunisieTelecom Agent ---
class EnhancedTunisieTelecomAgent:
    """Enhanced Tunisie Telecom specific agent for answering questions"""
    
    def __init__(self):
        """Initialize the TT agent"""
        self.qna_model = EnhancedQnAModel(Config.CSV_FILE)
        self.text_processor = TextProcessor()
        self.web_manager = GoogleSearchManager()
        self.answer_generator = AnswerGenerator()
        self.ollama_client = OllamaClient()  # Initialize Ollama client
        
        logger.info("TunisieTelecom Agent initialized successfully")
    
    def _is_query_tt_related(self, query, similarity_score=0.0):
        """Determine if a query is related to Tunisie Telecom"""
        query_lower = self.text_processor.basic_clean(query)
        
        # Check for TT keywords
        for keyword in TT_KEYWORDS:
            if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
                logger.info(f"Query is TT-related (matched keyword: '{keyword}')")
                return True
        
        # Check for similarity to known questions
        if similarity_score > 0.30:  # Lower threshold for topic detection
            logger.info(f"Query is TT-related (similarity to known question: {similarity_score:.2f})")
            return True
        
        logger.info(f"Query is not TT-related: {query}")
        return False
    
    def _is_query_off_topic(self, query):
        """Determine if a query is completely off-topic"""
        query_lower = self.text_processor.basic_clean(query)
        
        # Check for off-topic indicators
        for keyword in OFF_TOPIC_KEYWORDS:
            if keyword in query_lower and not any(tt_kw in query_lower for tt_kw in TT_KEYWORDS):
                logger.info(f"Query is off-topic (matched off-topic keyword: '{keyword}')")
                return True
        
        # Check for food-related queries
        food_terms = ["pizza", "recipe", "food", "cook", "restaurant", "meal", "dish"]
        if any(term in query_lower for term in food_terms) and not any(tt_kw in query_lower for tt_kw in TT_KEYWORDS):
            logger.info(f"Query is off-topic (food related)")
            return True
            
        return False
    
    def facebook_search(self, query):
        """Search Facebook public pages for relevant posts (placeholder: Google search with site:facebook.com)"""
        # In production, use Facebook Graph API or a proper scraper
        fb_results = self.web_manager.google_search(f"{query} Tunisie Telecom", site_filter="facebook.com", num_results=Config.MAX_WEB_RESULTS)
        fb_contents = []
        query_terms = self.text_processor.basic_clean(query).split()
        for result in fb_results:
            url = result.get('href')
            title = result.get('title', '')
            content = self.web_manager.fetch_web_content(url, query_terms)
            if content:
                fb_contents.append({
                    'url': url,
                    'title': title,
                    'content': content,
                    'source_type': 'Facebook',
                    'snippet': result.get('snippet', '')
                })
            elif result.get('snippet'):
                fb_contents.append({
                    'url': url,
                    'title': title,
                    'content': result['snippet'],
                    'source_type': 'Facebook',
                    'snippet': result['snippet']
                })
        return fb_contents[:Config.MAX_WEB_RESULTS]

    def get_llm_answer(self, query, use_ollama=True):
        """Get answer from LLM (Ollama or Grok) with system prompt to act as TT assistant"""
        system_prompt = "You are a helpful virtual assistant for Tunisie Telecom. Answer as a TT expert, using clear, concise, and friendly language."
        
        # Try Ollama first if enabled
        if use_ollama and Config.OLLAMA_ENABLED:
            try:
                logger.info("Attempting to use Ollama for response generation")
                if self.ollama_client.is_available():
                    answer = self.ollama_client.generate(query, system_prompt=system_prompt)
                    if answer and len(answer) > 10:  # Simple check to verify we got a real answer
                        logger.info("Successfully generated response with Ollama")
                        return answer
                    else:
                        logger.warning("Ollama returned empty or short response, falling back to Grok")
                else:
                    logger.warning("Ollama service is not available, falling back to Grok")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
                logger.info("Falling back to Grok API after Ollama error")
        
        # Fall back to Grok API
        GROK_API_KEY = os.environ.get("GROK_API_KEY") or "xai-C7tLsZEkKTYtauhZNBFKmPV5tLRraB5JZpGL1l3f3HdcKmSM3gsOuawcVKroeHfkaY5hd0NterLxDFEv"
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7
        }
        try:
            logger.info("Calling Grok API for response")
            resp = requests.post(url, headers={
                "Authorization": f"Bearer {GROK_API_KEY}",
                "Content-Type": "application/json"
            }, json=payload, timeout=15)
            if resp.ok:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", None)
        except Exception as e:
            logger.error(f"Grok API error: {e}")
        
        return None
        
    def grok_fallback(self, query):
        """Legacy method for backward compatibility"""
        return self.get_llm_answer(query)

    def process_query(self, query):
        logger.info(f"Processing query: {query}")
        if self._is_query_off_topic(query):
            return {
                'answer': "I specialize in Tunisie Telecom topics. Please ask me about Tunisie Telecom services, products, or support.",
                'source': "OFF_TOPIC",
                'matched_question': None,
                'confidence': 0,
                'informative': False,
                'formatted_answer': "I specialize in Tunisie Telecom topics. Please ask me about Tunisie Telecom services, products, or support."
            }
        # 1. Try local KB (CSV)
        local_result = self.qna_model.get_answer_with_details(query)
        similarity = local_result.get('similarity', 0)
        logger.info(f"Local match similarity: {similarity:.4f}")
        if local_result.get('matched_question'):
            logger.info(f"Best matched question: {local_result.get('matched_question')}")
        is_tt_query = self._is_query_tt_related(query, similarity)
        # 2. If not high/moderate confidence, try Google (official/general)
        web_results = []
        if is_tt_query and similarity < Config.LOCAL_MODERATE_CONFIDENCE and Config.ENABLE_WEB_SEARCH:
            logger.info("Performing Google web search for additional information")
            web_results = self.web_manager.search_web(query)
        # 3. If still not confident, try Facebook search
        fb_results = []
        if is_tt_query and similarity < Config.LOCAL_MODERATE_CONFIDENCE and not web_results:
            logger.info("Performing Facebook search for additional information")
            fb_results = self.facebook_search(query)
        # 4. Combine sources: CSV > Google > Facebook
        result = self.answer_generator.combine_sources(local_result, web_results or fb_results)
        result['matched_question'] = local_result.get('matched_question')
        result['query'] = query
        result['is_tt_related'] = is_tt_query
        # 5. If still no informative answer, try Ollama or Grok API
        if (not result.get('informative')) or (result.get('source') == 'TT_NOT_IN_KB_NO_WEB'):
            logger.info("Falling back to LLM (Ollama/Grok) for answer")
            
            # First try with Ollama
            llm_answer = self.get_llm_answer(query, use_ollama=True)
            source = 'OLLAMA' if Config.OLLAMA_ENABLED and self.ollama_client.is_available() else 'GROK_API'
            
            if llm_answer:
                result['answer'] = llm_answer
                result['source'] = source
                result['informative'] = True
                logger.info(f"Used {source} to generate response")
                result['formatted_answer'] = llm_answer
        # Format the final answer
        if result.get('answer'):
            result['formatted_answer'] = result['answer']
        else:
            result['formatted_answer'] = "Je n'ai pas d'information sur ce sujet."
        return result
    
    def get_answer(self, query):
        """Get a clean answer to a user query without source attribution"""
        result = self.process_query(query)
        
        # If it's an off-topic query, return the standard response
        if result.get('source') == "OFF_TOPIC":
            return "Je me spécialise dans les sujets Tunisie Telecom. Veuillez me poser des questions sur les services, produits ou support de Tunisie Telecom."
        
        # Extract just the core answer information without any formatting or attribution
        if result.get('answer'):
            answer_text = result['answer']
            
            # Remove any formatting elements or prefixes
            answer_text = re.sub(r'\[.*?\]:', '', answer_text).strip()
            
            # For specific queries about codes, ensure they're prominently shown
            if any(term in query.lower() for term in ['code', 'ussd', 'recharge', 'activation', 'activer']):
                # Extract potential code patterns like *123#, *123*xyz#
                code_patterns = re.findall(r'(\*\d+(?:\*[^*#]*?)?#)', answer_text)
                if code_patterns:
                    return code_patterns[0]  # Return the first code found
                
                # Look for patterns like "composez le *123#" or "tapez *123#"
                instruction_match = re.search(r'(?:composez|tapez|composer|taper|saisissez|utiliser)(?:\s+le)?\s+(\*\d+(?:\*[^*#]*?)?#)', answer_text, re.IGNORECASE)
                if instruction_match:
                    return instruction_match.group(1)
            
            # Summarize the answer in a concise way (1-2 sentences)
            return self.answer_generator.summarize_text(answer_text, num_sentences=2).strip()
        else:
            return "Je n'ai pas d'information spécifique sur ce sujet. Veuillez consulter le site officiel de Tunisie Telecom."

# --- Evaluation Utilities ---
class EvaluationManager:
    """Utilities for evaluating agent performance"""
    
    def __init__(self, agent):
        """Initialize with the agent to evaluate"""
        self.agent = agent
        self.evaluation_results = []
    
    def evaluate_on_test_set(self, test_set):
        """Evaluate the agent on a test set"""
        logger.info(f"Starting evaluation on {len(test_set)} test cases")
        
        correct = 0
        total = len(test_set)
        results = []
        
        for i, test_case in enumerate(test_set):
            query = test_case["query"]
            expected_source_category = test_case["expected_source_category"]
            expected_csv_question_match = test_case.get("expected_csv_question_match")
            
            # Process the query
            result = self.agent.process_query(query)
            actual_source = result.get('source', 'UNKNOWN')
            actual_matched_question = result.get('matched_question')
            
            # Determine correctness
            is_correct = False
            
            if actual_source == expected_source_category:
                if expected_source_category.startswith("LOCAL_") and expected_csv_question_match:
                    if actual_matched_question == expected_csv_question_match:
                        is_correct = True
                    else:
                        logger.info(f"CSV match incorrect. Expected: '{expected_csv_question_match}', Got: '{actual_matched_question}'")
                elif expected_source_category == "WEB_TT_FOUND":
                    if result.get('answer') and result.get('informative'):
                        is_correct = True
                    else:
                        logger.info(f"Web result not informative: {result.get('answer', '')[:100]}")
                else:
                    is_correct = True  # For OFF_TOPIC, etc.
            else:
                logger.info(f"Source category mismatch. Expected: {expected_source_category}, Got: {actual_source}")
            
            if is_correct:
                correct += 1
            
            # Record result
            results.append({
                'query': query,
                'expected_source': expected_source_category,
                'expected_matched_question': expected_csv_question_match,
                'actual_source': actual_source,
                'actual_matched_question': actual_matched_question,
                'answer': result.get('formatted_answer', "No answer generated"),
                'confidence': result.get('confidence', 0),
                'is_correct': is_correct
            })
        
        # Calculate metrics
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Calculate accuracy by category
        category_metrics = {}
        for result in results:
            expected_src = result['expected_source']
            if expected_src not in category_metrics:
                category_metrics[expected_src] = {'correct': 0, 'total': 0}
            
            category_metrics[expected_src]['total'] += 1
            if result['is_correct']:
                category_metrics[expected_src]['correct'] += 1
        
        for cat, metrics in category_metrics.items():
            metrics['accuracy'] = (metrics['correct'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
        
        # Store results
        self.evaluation_results = results
        
        # Return summary
        return {
            'total_cases': total,
            'correct_cases': correct,
            'overall_accuracy': accuracy,
            'category_metrics': category_metrics,
            'results': results
        }
    
    def print_evaluation_summary(self, results=None):
        """Print a summary of evaluation results"""
        if results is None and not self.evaluation_results:
            logger.error("No evaluation results to summarize")
            return
            
        if results is None:
            # Create summary from stored results
            correct_count = sum(1 for r in self.evaluation_results if r['is_correct'])
            total_count = len(self.evaluation_results)
            overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
            
            # Group by expected source
            category_metrics = {}
            for r in self.evaluation_results:
                src = r['expected_source']
                if src not in category_metrics:
                    category_metrics[src] = {'correct': 0, 'total': 0}
                
                category_metrics[src]['total'] += 1
                if r['is_correct']:
                    category_metrics[src]['correct'] += 1
            
            for cat, metrics in category_metrics.items():
                metrics['accuracy'] = (metrics['correct'] / metrics['total']) * 100 if metrics['total'] > 0 else 0
                
            results = {
                'total_cases': total_count,
                'correct_cases': correct_count,
                'overall_accuracy': overall_accuracy,
                'category_metrics': category_metrics,
                'results': self.evaluation_results
            }
        
        # Print summary information
        print("\n===== EVALUATION SUMMARY =====")
        print(f"Total test cases: {results['total_cases']}")
        print(f"Correct predictions: {results['correct_cases']}")
        print(f"Overall accuracy: {results['overall_accuracy']:.2f}%")
        
        print("\n----- Performance by Category -----")
        for category, metrics in results['category_metrics'].items():
            print(f"Category: {category}")
            print(f"  Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")

# --- Interactive CLI ---
def run_interactive_mode(agent):
    """Run an interactive Q&A session"""
    print("\n===== Tunisie Telecom Q&A System =====")
    print("Ask questions about Tunisie Telecom services, products, or support.")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print("=====================================\n")
    
    while True:
        try:
            # Get user input
            user_input = input("\nAsk about Tunisie Telecom: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Thank you for using the Tunisie Telecom Q&A system. Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Get direct answer without source attribution
            start_time = time.time()
            answer = agent.get_answer(user_input)
            end_time = time.time()
            
            # Display only the answer without prefixes
            print(f"\nAgent Answer: {answer}")
            
            # Only for debugging, can be removed for production
            print(f"\n(Query processed in {end_time - start_time:.2f} seconds)")
        
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting...")
            break
        
        except Exception as e:
            logger.error(f"Error in interactive mode: {str(e)}")
            print(f"An error occurred: {str(e)}")
# --- FastAPI App ---
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Tunisie Telecom Q&A API", description="Ask questions about Tunisie Telecom.")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agent (singleton)
agent = EnhancedTunisieTelecomAgent()

class AskRequest(BaseModel):
    question: str
    language: Optional[str] = "fr"

class AskResponse(BaseModel):
    answer: str
    matched_question: Optional[str] = None
    confidence: Optional[float] = None
    informative: Optional[bool] = None
    source: Optional[str] = None

# Import file handler
from fastapi import UploadFile, File, Form, HTTPException
from typing import List
from file_handler import FileUploadHandler

file_handler = FileUploadHandler(upload_dir="./uploads")

class AskRequest(BaseModel):
    question: str
    language: Optional[str] = "fr"
    use_ollama: Optional[bool] = True

@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    result = agent.process_query(req.question)
    
    # If we need to fallback to LLM, respect the use_ollama preference
    if (not result.get('informative')) or (result.get('source') == 'TT_NOT_IN_KB_NO_WEB'):
        use_ollama = req.use_ollama if req.use_ollama is not None else True
        llm_answer = agent.get_llm_answer(req.question, use_ollama=use_ollama)
        
        if llm_answer:
            source = 'OLLAMA' if use_ollama and Config.OLLAMA_ENABLED and agent.ollama_client.is_available() else 'GROK_API'
            result['answer'] = llm_answer
            result['formatted_answer'] = llm_answer
            result['source'] = source
            result['informative'] = True
    
    return AskResponse(
        answer=result.get("formatted_answer", "No answer found."),
        matched_question=result.get("matched_question"), 
        confidence=result.get("confidence"),
        informative=result.get("informative"),
        source=result.get("source")
    )
    
@app.post("/ask_with_files")
async def ask_with_files(
    question: str = Form(...),
    language: str = Form("fr"),
    use_ollama: bool = Form(True),
    files: List[UploadFile] = File(None)
):
    try:
        # Process uploaded files
        file_results = await file_handler.process_uploads(files)
        
        # Add file context to question
        file_context = ""
        for file_result in file_results:
            if "content" in file_result and file_result["content"]:
                file_context += f"\nContent from {file_result['filename']}:\n{file_result['content'][:500]}...\n"
        
        # Process query with file context
        enhanced_question = f"{question}\n\nContext from uploaded files: {file_context}" if file_context else question
        result = agent.process_query(enhanced_question)
        
        # If we need to fallback to LLM, respect the use_ollama preference
        if (not result.get('informative')) or (result.get('source') == 'TT_NOT_IN_KB_NO_WEB'):
            llm_answer = agent.get_llm_answer(enhanced_question, use_ollama=use_ollama)
            
            if llm_answer:
                source = 'OLLAMA' if use_ollama and Config.OLLAMA_ENABLED and agent.ollama_client.is_available() else 'GROK_API'
                result['answer'] = llm_answer
                result['formatted_answer'] = llm_answer
                result['source'] = source
                result['informative'] = True
        
        return {
            "answer": result.get("formatted_answer", "No answer found."),
            "matched_question": result.get("matched_question"),
            "confidence": result.get("confidence"),
            "informative": result.get("informative"),
            "source": result.get("source"),
            "processed_files": [{"filename": f["filename"], "type": f.get("type", "unknown")} for f in file_results]
        }
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
def root():
    return {"message": "Tunisie Telecom Q&A API. Use POST /ask with a JSON body {question: ...}"}
    
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "model_loaded": bool(agent),
    }
    
@app.get("/stats")
def get_stats():
    """Get usage statistics"""
    # In a production app, you'd track usage stats
    return {
        "total_queries": 0,  # Placeholder
        "uptime": "0 days",  # Placeholder
        "top_questions": []  # Placeholder
    }

# --- Ollama Client ---
class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url=Config.OLLAMA_URL, model=Config.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self.logger = logging.getLogger("TT_QA.OllamaClient")
        self.logger.info(f"Ollama client initialized with model: {model}")
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                self.logger.error(f"Failed to get models. Status: {response.status_code}")
                return []
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama: {str(e)}")
            return []
    
    def is_available(self):
        """Check if Ollama service is available"""
        try:
            models = self.get_available_models()
            return len(models) > 0
        except:
            return False
            
    def generate(self, prompt, system_prompt=None, temperature=Config.OLLAMA_TEMPERATURE, max_tokens=Config.OLLAMA_MAX_TOKENS):
        """Generate response using Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
                
            self.logger.debug(f"Sending request to Ollama: {self.api_endpoint}")
            response = requests.post(self.api_endpoint, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "No response from Ollama")
            else:
                self.logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return f"Error: Failed to get response from Ollama (Status: {response.status_code})"
                
        except Exception as e:
            self.logger.error(f"Error calling Ollama API: {str(e)}")
            return f"Error connecting to Ollama: {str(e)}"

@app.get("/ollama/models")
async def get_ollama_models():
    """Get available Ollama models and current configuration"""
    try:
        # Check if Ollama is available and get models
        models = agent.ollama_client.get_available_models()
        
        return {
            "current_model": agent.ollama_client.model,
            "available_models": models,
            "is_enabled": Config.OLLAMA_ENABLED,
            "is_available": len(models) > 0,
            "stats": agent.ollama_client.get_stats()
        }
    except Exception as e:
        logger.error(f"Error getting Ollama models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")

@app.post("/ollama/model")
async def set_ollama_model(model: str):
    """Set the Ollama model to use"""
    try:
        # Verify model is available
        available_models = agent.ollama_client.get_available_models()
        if model not in available_models:
            # Try to pull the model if not found
            logger.info(f"Model '{model}' not found. Attempting to pull it from Ollama repository...")
            pull_result = agent.ollama_client.pull_model(model)
            
            if not pull_result.get("success", False):
                raise HTTPException(status_code=404, detail=f"Model '{model}' not available and could not be pulled. Options: {available_models}")
            
            # Refresh the model list
            available_models = agent.ollama_client.get_available_models()
            if model not in available_models:
                raise HTTPException(status_code=404, detail=f"Model pulled but still not available. Options: {available_models}")
        
        # Set the model
        agent.ollama_client.model = model
        logger.info(f"Set Ollama model to: {model}")
        
        # Get model info
        model_info = agent.ollama_client.get_model_info(model)
        
        return {
            "message": f"Ollama model set to: {model}",
            "current_model": model,
            "model_info": model_info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting Ollama model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting Ollama model: {str(e)}")

@app.post("/ollama/toggle")
async def toggle_ollama(enable: bool = True):
    """Enable or disable Ollama"""
    try:
        # Update configuration
        Config.OLLAMA_ENABLED = enable
        status = "enabled" if enable else "disabled"
        logger.info(f"Ollama {status}")
        
        # If enabling, check if Ollama is actually available
        is_available = False
        if enable:
            is_available = agent.ollama_client.is_available()
            if not is_available:
                logger.warning("Ollama was enabled but server is not reachable")
        
        return {
            "message": f"Ollama has been {status}",
            "is_enabled": enable,
            "is_available": is_available if enable else False
        }
    except Exception as e:
        logger.error(f"Error toggling Ollama: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error toggling Ollama: {str(e)}")

@app.post("/ollama/pull")
async def pull_ollama_model(model: str):
    """Pull a new model from the Ollama library"""
    try:
        # Verify Ollama is available
        if not agent.ollama_client.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Ollama server is not available. Please make sure it's running."
            )
        
        # Pull the model
        result = agent.ollama_client.pull_model(model)
        if result.get("success", False):
            return {
                "message": f"Model {model} pulled successfully",
                "success": True
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to pull model: {result.get('message', 'Unknown error')}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pulling Ollama model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error pulling Ollama model: {str(e)}")
