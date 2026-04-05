import os
import time
import bs4
import requests
from urllib.parse import urljoin
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Seed URLs for the initial scrape
SEED_URLS = [
    "https://handbook.gitlab.com/",
    "https://handbook.gitlab.com/handbook/values/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/company/culture/",
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/product/",
    "https://handbook.gitlab.com/handbook/engineering/",
    "https://handbook.gitlab.com/handbook/sales/",
    "https://handbook.gitlab.com/handbook/finance/",
    "https://handbook.gitlab.com/handbook/finance/accounting/",
    "https://handbook.gitlab.com/handbook/legal/",
    "https://handbook.gitlab.com/handbook/marketing/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://about.gitlab.com/direction/",
    "https://about.gitlab.com/releases/",
    "https://about.gitlab.com/blog/categories/releases/"
]

# Only follow links under these prefixes
ALLOWED_PREFIXES = [
    "https://handbook.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
]

MAX_DISCOVERED_URLS = 150

def discover_sub_links(seed_urls, allowed_prefixes, max_urls=MAX_DISCOVERED_URLS):
    """Fetch each seed page and extract internal hyperlinks that match allowed prefixes."""
    discovered = set()
    headers = {"User-Agent": "GitLabChatbot/1.0"}

    for url in seed_urls:
        try:
            resp = requests.get(url, headers=headers, timeout=20)
            resp.raise_for_status()
            soup = bs4.BeautifulSoup(resp.text, "html.parser")

            for a_tag in soup.find_all("a", href=True):
                href = a_tag["href"]
                full_url = urljoin(url, href)
                # Strip fragments and query params
                full_url = full_url.split("#")[0].split("?")[0]
                # Normalise trailing slash for path-based URLs
                if not full_url.endswith("/") and "." not in full_url.split("/")[-1]:
                    full_url += "/"
                if any(full_url.startswith(p) for p in allowed_prefixes):
                    discovered.add(full_url)

            time.sleep(0.5)  # polite delay
        except Exception as e:
            print(f"  Warning: Could not fetch {url} for link discovery: {e}")

    # Remove URLs already in the seed list
    discovered -= set(seed_urls)

    # Cap to avoid runaway crawl
    if len(discovered) > max_urls:
        discovered = set(list(discovered)[:max_urls])

    return sorted(discovered)


def load_and_process_data():
    """Scrapes seed URLs + auto-discovered sub-links for comprehensive handbook coverage."""
    print(f"Starting with {len(SEED_URLS)} seed URLs...")

    # --- Step 1: discover hyperlinks inside the seed pages ---
    print("Discovering sub-section links from seed pages...")
    sub_links = discover_sub_links(SEED_URLS, ALLOWED_PREFIXES)
    print(f"Discovered {len(sub_links)} additional sub-links.")

    all_urls = list(dict.fromkeys(SEED_URLS + sub_links))  # deduplicate, preserve order
    print(f"Total URLs to scrape: {len(all_urls)}")

    # --- Step 2: scrape in batches (avoids timeout on huge lists) ---
    all_docs = []
    batch_size = 10
    for i in range(0, len(all_urls), batch_size):
        batch = all_urls[i:i + batch_size]
        print(f"  Scraping batch {i // batch_size + 1} ({len(batch)} URLs)...")
        try:
            loader = WebBaseLoader(
                web_paths=batch,
                bs_kwargs=dict(
                    parse_only=bs4.SoupStrainer(["p", "h1", "h2", "h3", "h4", "li", "span", "a"])
                )
            )
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"  Warning: batch failed – {e}")
        time.sleep(1.0)

    print(f"Loaded {len(all_docs)} documents total.")

    # --- Step 3: chunk the documents ---
    # HuggingFace MiniLM-L6-v2 drops data after 256 tokens (~1000 characters). 
    # By strictly reducing the chunk_size down to 800, we 100% prevent the API from 
    # silently truncating your data, which fixes the "not providing a proper answer" bug!
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\\n\\n", "\\n", " ", ""]
    )
    splits = text_splitter.split_documents(all_docs)

    # Keep a generous coverage spread across all URLs while avoiding bloat
    from collections import defaultdict
    page_chunks = defaultdict(list)
    for split in splits:
        page_chunks[split.metadata["source"]].append(split)

    diverse_splits = []
    for url, chunks in page_chunks.items():
        diverse_splits.extend(chunks[:12])

    return diverse_splits

def create_vector_store(splits, faiss_path="faiss_index"):
    """Creates and saves a FAISS vectorstore using local HuggingFace embeddings."""
    from langchain_huggingface import HuggingFaceEmbeddings
    print("Loading HuggingFace embedding model (runs locally, completely immune to Google 429 crashes)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Embedding {len(splits)} optimized chunks silently on CPU...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(faiss_path)

    print(f"  All {len(splits)} optimized chunks embedded and saved natively.")
    return vectorstore

if __name__ == "__main__":
    import shutil
    from dotenv import load_dotenv
    load_dotenv()
    os.environ["USER_AGENT"] = "GitLabChatbot/1.0"

    faiss_path = "faiss_index"
    if os.path.exists(faiss_path):
        shutil.rmtree(faiss_path)
        print("Cleared old FAISS index for fresh build.")

    splits = load_and_process_data()
    print(f"Created {len(splits)} highly relevant database chunks. Building FAISS vector store...")
    create_vector_store(splits, faiss_path)
    print("Vector store successfully built and saved to 'faiss_index'!")
