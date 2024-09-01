from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel


def generate_tweet(llm, title, summary, sentiment, tone):

    print("Generating Tweet...")
    prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that specializes in writing tweets.
                    Your task is to write a tweet about the provided context, and capturing the indicated sentiment and tone.
                    If you're not sure about the sentiment or tone, just return an neutral response.
                    If the contet provided do not contain enough information just return an empty response, do not make up information.
                    """,
                ),
                ("human", "Context: {context}\n Sentiment: {sentiment}\n Tone: {tone}")
            ]
        )
    
    context = f"{title:}\n{summary}"

    tweet_chain = prompt_template | llm

    response = tweet_chain.invoke({
        "context": context,
        "sentiment": sentiment,
        "tone": tone
    })
    return response