import re
import time
import streamlit as st

from langchain_community.document_loaders import SeleniumURLLoader, UnstructuredURLLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.runnables import RunnableLambda


class SummmaryWithTitle(BaseModel):
    '''A summary and a title for the trending topic.'''
    title: str
    summary: str

class RAGTrends():

    def __init__(self, llm_ollama_model, hf_embeddings_model):
        
        self.llm = llm_ollama_model
        self.embeddings_model = hf_embeddings_model
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a helpful assistant that specializes in article summarization.
                    Your task is to summarize a given text article without refering it, and generate a title for it.
                    The summary should be between 4 and 10 lines, depending on how many details are given.
                    If the provided article doesn't contain coherent and meaningful content, just return an empty response.
                    """,
                ),
                ("human", "Article: {article}"),
            ]
        )
        self.dict_schema = convert_to_openai_tool(SummmaryWithTitle)

    
    def _url_loader(self, url_list:list[str], loader_type:str="Unstructured"):
        """
        Loads data from a list of URLs using the specified loader type.

        Args:
            url_list (list[str]): A list of URLs to load data from.
            loader_type (str, optional): The type of loader to use. Defaults to "Unstructured".
                Supported values are "Unstructured", "Selenium", and "Base".

        Returns:
            data: The loaded data.

        Raises:
            ValueError: If the specified loader type is not supported.
        """
        if loader_type == "Unstructured":
            unstruct_loader = UnstructuredURLLoader(url_list)
            data = unstruct_loader.load()
        elif loader_type == "Selenium":
            sele_loader = SeleniumURLLoader(url_list)
            driver = sele_loader._get_driver()
            try:
                data = sele_loader.load()
            finally:
                driver.close()
                driver.quit()
        elif loader_type == "Base":
            header_template = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            base_loader = WebBaseLoader(url_list, header_template=header_template)
            data = base_loader.load()
        else:
            raise ValueError(f"Loader type {loader_type} not supported.")
        return data
    
    def _rec_splitter(self, url_doc_list:list):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                    chunk_overlap=50,
                                                    add_start_index=True)
        all_splits = text_splitter.split_documents(url_doc_list)
        return all_splits

    def _reteived_docs_parser(self, ret_doc_list:list):
        """
        Parses a list of retrieved documents and extracts the meaningful sentences from the article.

        Args:
            ret_doc_list (list): A list of retrieved documents.

        Returns:
            str: The meaningful sentences from the article, joined by newline characters.

        Description:
            This function takes a list of retrieved documents and extracts the meaningful sentences from the scraped article text (not cleaned).
            It joins the page content of each document into a single string, replacing consecutive newline characters with a dot followed by a space.
            Then, it splits the article into sentences using a regular expression pattern.
            The function filters out sentences that have fewer than 5 words and joins the remaining sentences into a single string, separated by newline characters.
            The resulting string contains the meaningful sentences from the article.
        """
        ret_article = "\n".join([doc.page_content for doc in ret_doc_list])
        ret_article = ret_article.replace('\n\n', '. ')
        ret_article_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z][a-z][A-Z]\.)(?<=\.|\?|!|\n)\s*', ret_article)
        ret_article_meaningful_sentences = [s for s in ret_article_sentences if len(s.split()) > 5]
        meaningful_ret_article = '\n'.join(ret_article_meaningful_sentences)
        return meaningful_ret_article
    

    def _run_rag_chain_once(self, trend, google_df, ddg_df, reteived_docs_parser_runnable, structured_output_llm):
        """
        Runs a single iteration of the RAG pipeline for a given trend.

        Args:
            trend (str): The trend keyword.
            google_df (pandas.DataFrame): The Google Trends data.
            ddg_df (pandas.DataFrame): The DuckDuckGo news data.
            reteived_docs_parser_runnable (callable): The function to parse retrieved documents.
            structured_output_llm (callable): The function to generate structured output using an LLM.

        Returns:
            Any: The results of the RAG pipeline.

        Description:
            This function runs a single iteration of the RAG pipeline for a given trend. It starts by printing a message indicating the start of the pipeline.
            It then prints a message indicating the start of the scraping process. It retrieves the URL list for the given trend from the Google Trends data.
            It tries to load the URL documents using the "Base" loader. If the total length of the page content of the loaded documents is less than 5000 characters, it uses the "Unstructured" loader instead.
            It then checks if any of the loaded documents have an empty page content and their metadata source is present in the DuckDuckGo news data. If so, it appends the corresponding article body to the page content.
            It then prints a message indicating the creation of the FAISS vector store. It splits the loaded documents into smaller chunks using the `rec_splitter` method.
            It creates a FAISS vector store from the split documents using the `embeddings_model`. It creates a retriever based on the FAISS vector store.
            It retrieves the query from the Google Trends data for the given trend. It then prints a message indicating the start of the RAG process.
            It invokes the RAG chain by passing the query to the chain. The chain consists of the FAISS retriever, the retrieved documents parser, the prompt template, and the structured output LLM.
            It returns the results of the RAG pipeline.

        Note:
            - This function assumes that the necessary dependencies and objects are already imported and initialized.
            - The `url_loader` method is assumed to be a method of the current class.
            - The `rec_splitter` method is assumed to be a method of the current class.
            - The `prompt_template` and `structured_output_llm` objects are assumed to be initialized and passed as arguments.
            - The `embeddings_model` object is assumed to be initialized and passed as an argument.
        """
        
        print(f"# Starting RAG pipeline for heyword: {trend}")
        start = time.time()

        print("## Scraping Articles")
        df_trend = google_df[google_df["trend_kws"]==trend]
        url_list = df_trend['url'].iloc[0]
        try:
            url_docs = self._url_loader(url_list, "Base")
            if len(" ".join([doc.page_content for doc in url_docs])) < 5000:
                print("Using UnstructuredLoader")
                url_docs = self._url_loader(url_list, "Unstructured")
        except Exception as e:
            print("Exception BaseURLLoader:", e)
            print("Using UnstructuredLoader")
            url_docs = self._url_loader(url_list, "Unstructured")

            
        for doc in url_docs:
            if (doc.page_content == "") & (doc.metadata["source"] in ddg_df.url.to_list()):
                article_body_index = ddg_df['url'].to_list().index(doc.metadata["source"])
                doc.page_content += ddg_df['body'][article_body_index]
        scraping_checkpoint = time.time()
        scraping_dur = scraping_checkpoint - start

        print("## Creating FAISS vector store")
        splits_docs = self._rec_splitter(url_docs)
        faiss_db = FAISS.from_documents(splits_docs, self.embeddings_model)
        faiss_retriever = faiss_db.as_retriever(search_type="similarity",
                                    search_kwargs={'k': 5})
        ret_query = '\n'.join(df_trend['title'].iloc[0])
        faiss_checkpoint = time.time()
        faiss_dur = faiss_checkpoint - scraping_checkpoint

        print("## Performing RAG")
        rag_chain = (faiss_retriever
                        | { "article": reteived_docs_parser_runnable }
                        | self.prompt_template
                        | structured_output_llm)
        rag_results = rag_chain.invoke(ret_query)
        end = time.time()
        chain_dur = end - faiss_checkpoint
        print(f"Scrape: {scraping_dur}, Faiss: {faiss_dur}, Chain: {chain_dur}")
        return rag_results


    def run_rag(self, google_df, ddg_df):
        """
        Runs the RAG (Retrieval-based Article Generation) process for each trend keyword in the given Google DataFrame.

        Args:
            google_df (DataFrame): The Google DataFrame containing trend keywords and their corresponding URLs.
            ddg_df (DataFrame): The DataFrame containing DuckDuckGo search results.

        Returns:
            dict: A dictionary containing the trend keywords, titles, and summaries generated by the RAG process.
                - "Trend_kws" (list): A list of trend keywords.
                - "Title" (list): A list of generated titles.
                - "Summary" (list): A list of generated summaries.
        """

        results={"Trend_kws":[], "Title":[], "Summary":[]}
        trend_kws = google_df.trend_kws.to_list()
        reteived_docs_parser_runnable = RunnableLambda(self._reteived_docs_parser)
        structured_output_llm = self.llm.with_structured_output(self.dict_schema)

        for trend_kw in trend_kws:
            if google_df[google_df["trend_kws"]==trend_kw]['url'].iloc[0]:
                rag_results = self._run_rag_chain_once(trend_kw, google_df, ddg_df, reteived_docs_parser_runnable, structured_output_llm)
            else:
                rag_results = {'title':trend_kw, 'summary':'No enough information yet!'}
            print(trend_kw, ' ### ', rag_results['title'], ' ### ', rag_results['summary'], '\n\n')
            results['Trend_kws'].append(trend_kw)
            results['Title'].append(rag_results['title'])
            results['Summary'].append(rag_results['summary'])
            
        return results
