import pandas as pd
from dotenv import load_dotenv
import os
import openai
from scipy import spatial
import ast
import tiktoken

#set up OpenAi client instance
client = openai.OpenAI()

#load csv of text and embeddings from data/papers.csv
df = pd.read_csv("data/papers.csv")
df["embedding"] = df["embedding"].apply(ast.literal_eval) #cast str type back to list

# set up functions for search and ask, that will be used in the question-answer-app

# search function takes a user prompt, embedds it and returns those texts with the highest relevance and the relevance scores
def papers_by_relevance(
    query: str,
    df: pd.DataFrame,
    relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x,y),
    top_n: int = 50
) -> tuple[list[str], list[float]]:

    """Returns a list of texts and of relevance scores sorted from most related to least"""
    # get embedding of user prompt
    response = client.embeddings.create(model="text-embedding-ada-002", input= [query])
    resp_embed = response.data[0].embedding

    #get text and relatedness score for each row in df
    text_and_relatedness = []

    for i, row in df.iterrows():
        txt_rel = (row["text"], relatedness_fn(resp_embed, row["embedding"]))
        text_and_relatedness.append(txt_rel)

    #sort list of tuples by second element in tuple and descending
    text_and_relatedness.sort(key=lambda x: x[1], reverse=True)

    #zip through tuples to make separate lists
    texts, relatedness = zip(*text_and_relatedness)

    return texts[:top_n], relatedness[:top_n]

# Ask function takes the most relevant papers to user input and feeds these as additional
# info to answer the question, returns ChatGPT's response

# requires some sub-functions: counting tokens and generating the complete query message

def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    texts, relatednesses = papers_by_relevance(query, df)
    introduction = 'Use the following summaries of scientific papers to answer the question at the bottom. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for text in texts:
        next_article = f'\n\nNext paper summary:\n"""\n{text}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question

# main ask function
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = "gpt-3.5-turbo",
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:

    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""

    message = query_message(query, df, model=model, token_budget=token_budget)

    if print_message:
        print(message)

    #set up messages for the chat api call
    messages = [
        {"role": "system", "content": "You answer questions about the LLM Lama-2 by Meta."},
        {"role": "user", "content": message},
    ]

    # call
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content

    return response_message

# for aesthetic reasons: a visual buffer to use when necesary
buffer = "\n----------------------------------------------------\n\n"

# Basic program structure infinite while loop asking for a prompt from user
# keyword "quit" as input will cause a break in the loop and exit the program

while True:
    prompt = input("\n What do you want to know about Llama-2? \n\n To exit, type 'quit' and hit enter. \n\n"+buffer )

    if prompt == "quit":
        break

    # all functions are nested within the main ask function
    resp_message = ask(query=prompt, df=df)

    #print response message from chat and add several new lines of dashes to
    #separate message from next input prompt

    print("\n" + resp_message + "\n" + buffer + buffer + buffer)
