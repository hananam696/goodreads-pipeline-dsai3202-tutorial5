############################################################
# STEP 0: Setup and Packages
############################################################

# Standard libraries for system operations, package installation, and memory management
import sys
import subprocess
import os
import pandas as pd
import gc  # garbage collector to free memory

# Function to install required Python packages if not already available
def pip_install(pkgs):
    print(f"Installing packages: {pkgs}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *pkgs])
    print("All requested pip installs finished!", flush=True)

# List of required packages for this notebook/script
pkgs = [
    "boto3", "s3fs", "textstat", "nltk", "transformers==4.44.2", "torch==2.2.2",
    "pyarrow", "awswrangler", "numpy==1.24.1", "pandas==1.1.3", "python-dateutil==2.8.1",
    "textblob", "scikit-learn"
]

# Install all the packages
pip_install(pkgs)

# NLTK library for natural language processing
import nltk
# Download the VADER lexicon for sentiment analysis
nltk.download("vader_lexicon")

print("------ ENV BOOTSTRAP SUCCESS ------", flush=True)

############################################################
# STEP 1: Parquet Reading
############################################################

# Function to recursively find all parquet files in a directory
def find_parquet_files(directory):
    parquet_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet") or file.endswith(".parquet.snappy"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

# Set input and output directories using environment variables or default paths
INPUT_DIR = os.getenv("INPUT_DIR", "/opt/ml/processing/input/features/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/opt/ml/processing/output/")
os.makedirs(OUTPUT_DIR, exist_ok=True)  # create output directory if not exists

# Display the mounted directories and files for debugging
print(f"Searching recursively for Parquet files in: {INPUT_DIR}", flush=True)
for root, dirs, files in os.walk(INPUT_DIR):
    print(f"Mounted: {root}\n  Dirs: {dirs}\n  Files: {files}")

# Get all parquet files from the input directory
parquet_files = find_parquet_files(INPUT_DIR)
if not parquet_files:
    raise FileNotFoundError("No .parquet files found!")
print(f"Found parquet files: {parquet_files}")

############################################################
# STEP 2: Feature Extraction Functions
############################################################

# Import additional libraries for feature extraction
import glob, uuid, re, string, traceback
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from textblob import TextBlob

# Import HuggingFace Transformers and PyTorch for embeddings and zero-shot classification
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
except Exception as e:
    print("FATAL: transformers/torch not available in the container.", flush=True)
    raise

# Hyperparameters for text processing and batching
MAX_LEN = int(os.getenv("MAX_LEN", "96"))        # Max token length for embedding
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for embedding
ROW_CHUNK = int(os.getenv("ROW_CHUNK", "20000")) # Rows per chunk for processing
HOST = os.getenv("SM_CURRENT_HOST", "host0")     # Hostname for distributed processing

# Setup HuggingFace cache directory
os.environ.setdefault("HF_HOME", "/opt/ml/processing/hf-cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# Limit PyTorch threads for faster CPU usage
try:
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "2"))))
except Exception:
    pass

# Regular expressions for word and emoji detection
WORD_RE = re.compile(r"\w+")
EMOJI_RE = re.compile("[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
                      "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
                      "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FA6F" "\U0001FA70-\U0001FAFF"
                      "\u2600-\u26FF" "\u2700-\u27BF" "]", flags=re.UNICODE)

# Define required columns for the dataset
REQUIRED_COLUMNS = [
    "book_id", "user_id", "review_id", "rating", "rating_count", "review_text",
    "review_length_raw", "review_char_count", "date_added", "n_votes", "title",
    "publication_year", "author_names"
]

# Function to ensure all required columns exist
def filter_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    for col in missing:
        df[col] = pd.NA  # add missing columns as empty
    filtered = df[REQUIRED_COLUMNS]
    print(f"DF shape after column filter: {filtered.shape} (rows, columns)", flush=True)
    return filtered

############################################################
# STEP 3: Feature Engineering Functions
############################################################

# Text formatting features (sentence, paragraph, quotes, punctuation)
def formatting_features(df, txt):
    df["sentence_count"] = txt.apply(lambda s: s.count('.') + s.count('!') + s.count('?')).astype("int32")
    df["paragraph_count"] = txt.apply(lambda s: s.count('\n')+1).astype("int32")
    df["quote_count"] = txt.apply(lambda s: s.count('"')).astype("int32")
    df["exclamation_count"] = txt.apply(lambda s: s.count('!')).astype("int32")
    df["question_count"] = txt.apply(lambda s: s.count('?')).astype("int32")

# Count punctuation, uppercase letters, and emojis
def length_features(df, txt):
    df["punct_count"] = txt.apply(lambda s: sum(1 for ch in s if ch in string.punctuation)).astype("int32")
    df["caps_count"] = txt.apply(lambda s: sum(1 for ch in s if ch.isupper())).astype("int32")
    df["emoji_count"] = txt.apply(lambda s: len(EMOJI_RE.findall(s))).astype("int32")

# Lexical diversity (ratio of unique words to total words)
def lexical_features(df, txt):
    diversity = txt.apply(lambda s: len(set(s.split()))/max(1, len(s.split())))
    df["lexical_diversity"] = diversity.astype("float32")

# Readability using Flesch-Kincaid grade
def readability_features(df, txt):
    def flesch_kincaid(text):
        if not isinstance(text, str) or not text.strip():
            return 0.0
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        words = WORD_RE.findall(text)
        if not sentences or not words:
            return 0.0
        syllables = sum(len(re.findall(r"[aeiouyAEIOUY]+", w)) for w in words)
        word_count, sentence_count = len(words), max(len(sentences), 1)
        grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllables / word_count) - 15.59
        return float(grade) if grade > 0 else 0.0
    df["readability_grade"] = txt.apply(flesch_kincaid).astype("float32")

# Sentiment features using VADER and TextBlob
def sentiment_features(df, txt):
    sia = SentimentIntensityAnalyzer()
    sent = txt.apply(sia.polarity_scores)
    df["vader_pos"] = sent.apply(lambda d: d["pos"]).astype("float32")
    df["vader_neg"] = sent.apply(lambda d: d["neg"]).astype("float32")
    df["vader_neu"] = sent.apply(lambda d: d["neu"]).astype("float32")
    df["blob_subjectivity"] = txt.apply(lambda s: TextBlob(s).subjectivity).astype("float32")
    df["blob_polarity"] = txt.apply(lambda s: TextBlob(s).polarity).astype("float32")

# TF-IDF features (mean, max, min)
def tfidf_features(df, txt):
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    docs = txt.fillna("").tolist()
    X_tfidf = tfidf_vectorizer.fit_transform(docs)
    df["tfidf_mean"] = np.asarray(X_tfidf.mean(axis=1)).ravel()
    df["tfidf_max"] = np.asarray(X_tfidf.max(axis=1).toarray()).ravel()
    df["tfidf_min"] = np.asarray(X_tfidf.min(axis=1).toarray()).ravel()

# Topic modeling with NMF (extract top 2 topics)
def topic_features(df, txt):
    tfidf_vectorizer = TfidfVectorizer(max_features=300)
    nmf_model = NMF(n_components=5, init="nndsvda", random_state=42)
    docs = txt.fillna("").tolist()
    X_tfidf = tfidf_vectorizer.fit_transform(docs)
    nmf_topics = nmf_model.fit_transform(X_tfidf)
    df["topic1_prob"] = nmf_topics[:,0]
    df["topic2_prob"] = nmf_topics[:,1]

# Zero-shot classification using HuggingFace pipeline
def zeroshot_features(df, txt):
    zeroshot_pipe = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=-1)
    docs = txt.fillna("").tolist()
    candidate_labels = ["Contains spoilers", "Recommendation", "Critical", "Summary", "Praise", "Complaint"]
    results = zeroshot_pipe(docs, candidate_labels, multi_label=True)
    df["zeroshot_label"] = [','.join([lbl for lbl, score in zip(r['labels'], r['scores']) if score > 0.5]) for r in results]
    df["zeroshot_score"] = [max(r['scores']) if r['scores'] else 0.0 for r in results]

# Embedding features using DistilBERT
def embedding_features(df, txt):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased").eval()
    device = torch.device("cpu")
    model.to(device)
    embs = []
    docs = txt.tolist()
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i:i+BATCH_SIZE]
        with torch.no_grad():
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt"
            )
            out = model(**enc).last_hidden_state
            cls = out[:, 0, :].numpy()
            embs.append(cls)
    emb_all = np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)
    df["distilbert_embedding"] = [row.astype(np.float32).tolist() for row in emb_all]

############################################################
# STEP 4: Process Each Chunk of Data
############################################################

def process_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    txt = df_chunk["review_text"].fillna("")
    print("STEP 1: Formatting & Length Features")
    formatting_features(df_chunk, txt)
    length_features(df_chunk, txt)
    print("STEP 2: Lexical Features")
    lexical_features(df_chunk, txt)
    print("STEP 3: Readability Features")
    readability_features(df_chunk, txt)
    print("STEP 4: Sentiment Features")
    sentiment_features(df_chunk, txt)
    print("STEP 5: TFIDF Features")
    tfidf_features(df_chunk, txt)
    print("STEP 6: Topic Model Features")
    topic_features(df_chunk, txt)
    print("STEP 7: Zero-shot Classification Features")
    zeroshot_features(df_chunk, txt)
    print("STEP 8: Embedding Features")
    embedding_features(df_chunk, txt)
    print(f"DF shape immediately before write_out: {df_chunk.shape} (rows, columns)", flush=True)
    return df_chunk

############################################################
# STEP 5: Write Output Parquet
############################################################

def write_out(df_chunk: pd.DataFrame, base_name: str):
    if df_chunk.empty:
        print(f"write_out skipped: DataFrame is empty!", flush=True)
        return
    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
    out_path = os.path.join(
        OUTPUT_DIR, f"{base_name}_{HOST}_{uuid.uuid4().hex[:8]}.parquet"
    )
    pq.write_table(table, out_path, compression="snappy")
    print(f"write_out completed. Output path: '{out_path}' -- shape: {df_chunk.shape}", flush=True)

############################################################
# STEP 6: Main Workflow (Trial on first 500 rows)
############################################################

def main():
    try:
        # Print environment info for debugging
        print("Python:", sys.version)
        print("Pandas:", pd.__version__)
        print("NumPy:", np.__version__)
        try:
            import transformers as _tf, torch as _th
            print("Transformers:", _tf.__version__, "Torch:", _th.__version__)
        except Exception as _e:
            print("Transformers/Torch not importable:", _e)

        # Read the first parquet file as a trial
        p = parquet_files[0]
        print(f"Processing trial parquet: {p}")
        df = pd.read_parquet(p)
        print(f"DF shape at initial read: {df.shape} (rows, columns)", flush=True)

        # Filter columns to keep only required ones
        print("---- Filtering for specific columns ----")
        df_trial = df.head()  # trial on first 5 rows
        df_trial = filter_columns(df_trial)
        print(f"Columns after filtering: {list(df_trial.columns)}", flush=True)

        # Apply feature engineering pipeline
        print("---- Feature Engineering Pipeline ----")
        out_df = process_chunk(df_trial)

        # Write output parquet
        write_out(out_df, base_name="features_enriched_trial500")
        print("------ PROCESSING AND EMBEDDING SUCCESS ------", flush=True)

        # Free memory
        del out_df, df_trial, df
        gc.collect()

    except Exception as e:
        print("FATAL ERROR:", e, flush=True)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
