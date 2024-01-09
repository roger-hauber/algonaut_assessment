import pandas as pd
import openai
import requests
from dotenv import load_dotenv
import os
import urllib
import xml.etree.ElementTree as ET
import string
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords



# function to get paper titles and summaries from ArXiv
def fetch_papers():

    """Fetches papers from the arXiv API and returns them as a list of strings."""

    url = 'http://export.arxiv.org/api/query?search_query=ti:llama&start=0&max_results=70'

    response = urllib.request.urlopen(url)

    data = response.read().decode('utf-8')

    root = ET.fromstring(data)

    papers_list = []

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):

        title = entry.find('{http://www.w3.org/2005/Atom}title').text

        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text

        paper_info = f"Title: {title}\nSummary: {summary}\n"

        papers_list.append(paper_info)

    return papers_list

#function to clean up text data
def cleaning(text: str) -> str:
    """
    Return cleaned up version of input string: lowercase, replacing special characters, removing stopwords,
    and stemming
    """
    #lowercase
    clean_text = text.lower()
    #strip white space
    clean_text = clean_text.strip()
    #remove punctuation
    clean_text = "".join([letter for letter in clean_text if letter not in string.punctuation])
    #remove stopwords
    stop_words = set(stopwords.words("english"))
    clean_text = [word for word in word_tokenize(clean_text) if not word.lower() in stop_words]
    # stemming/lemmatizing
    clean_text = " ".join([WordNetLemmatizer().lemmatize(word) for word in clean_text])

    return clean_text



##
papers_list = fetch_papers()

#apply cleaning function
papers_list_clean = [cleaning(text) for text in papers_list]


# get embeddings for each paper in papers_list
EMBEDDING_MODEL = "text-embedding-ada-002" #specifying which embedding model to use
#set up OpenAI client instance
client = openai.OpenAI()

embedds = []


#get api response for pap in papers_list and append it to embeddings
for pap in papers_list_clean:
    pap = pap.replace("\n", " ")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input = [pap])

    # embedding list is nested in response.data[0].embedding
    embedds.append(response.data[0].embedding)


# get both original text and embedding into a dataframe
df = pd.DataFrame({"text": papers_list, "embedding": embedds})

# save dataframe to csv (for now)
df.to_csv("data/papers.csv")
