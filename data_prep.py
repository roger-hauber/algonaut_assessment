import pandas as pd
import openai
import requests
from dotenv import load_dotenv
import os
import urllib
import xml.etree.ElementTree as ET

#load dotenv and get openai key
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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

##
papers_list = fetch_papers()

# get embeddings for each paper in papers_list
EMBEDDING_MODEL = "text-embedding-ada-002" #specifying which embedding model to use
#set up OpenAI client instance
client = openai.OpenAI()

embedds = []


#get api response for pap in papers_list and append it to embeddings
for pap in papers_list:
    pap = pap.replace("\n", " ")
    response = client.embeddings.create(model=EMBEDDING_MODEL, input = [pap])

    # embedding list is nested in response.data[0].embedding
    embedds.append(response.data[0].embedding)


# get both original text and embedding into a dataframe
df = pd.DataFrame({"text": papers_list, "embedding": embedds})

# save dataframe to csv (for now)
df.to_csv("data/papers.csv")
