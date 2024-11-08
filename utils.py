import os

from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain.globals import set_llm_cache
from langchain_aws import ChatBedrock
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import re
import sqlite3
import pickle
import hashlib
from functools import wraps

from langchain_community.cache import SQLiteCache

load_dotenv()

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm = ChatBedrock(
    credentials_profile_name="default", model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
)



# llm = ChatGroq(
#     model="llama-3.1-70b-versatile",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )

def cache_to_sqlite(db_name='cache.db'):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Connect to the SQLite database
            conn = sqlite3.connect(db_name)
            c = conn.cursor()

            # Create the cache table if it doesn't exist
            c.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    func_module TEXT,
                    func_name TEXT,
                    key TEXT,
                    result BLOB,
                    PRIMARY KEY (func_module, func_name, key)
                )
            ''')

            # Create a unique key based on function arguments
            key = repr((args, sorted(kwargs.items())))
            key_hash = hashlib.sha256(key.encode('utf-8')).hexdigest()

            # Identify the function uniquely
            func_module = func.__module__
            func_name = func.__name__

            # Check if result is in cache
            c.execute('''
                SELECT result FROM cache WHERE func_module=? AND func_name=? AND key=?
            ''', (func_module, func_name, key_hash))
            row = c.fetchone()

            if row:
                # Load and return the cached result
                result = pickle.loads(row[0])
                conn.close()
                # print(f"Returning cached result for {func_name}{args}")
                return result
            else:
                # Compute the result and cache it
                result = func(*args, **kwargs)
                result_blob = pickle.dumps(result)
                c.execute('''
                    INSERT INTO cache (func_module, func_name, key, result)
                    VALUES (?, ?, ?, ?)
                ''', (func_module, func_name, key_hash, result_blob))
                conn.commit()
                conn.close()
                return result

        return wrapper

    return decorator


@cache_to_sqlite(db_name='./.firecrawl.db')
def get_web_content(url: str) -> str:
    app = FirecrawlApp(api_key=os.environ.get('FIRECRAWL_API_KEY'))
    response = app.scrape_url(url=url, params={
        'formats': ['markdown'],
        'onlyMainContent': False
    })
    return response['markdown']


def extract_protocol_and_domain(url):
    # Regular expression pattern to match protocol and domain
    pattern = r'^(https?://[^/]+)'
    match = re.match(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


if __name__ == "__main__":
    app = FirecrawlApp(api_key=os.environ.get('FIRECRAWL_API_KEY'))
    response = app.scrape_url(url='https://www.tweakers.net', params={
        'formats': ['markdown'],
        'onlyMainContent': False
    })
    print(response['markdown'])