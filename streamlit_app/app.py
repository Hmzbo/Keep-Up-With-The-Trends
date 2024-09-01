import re
import pandas as pd
import streamlit as st
from sympy import use
from scraper import TrendScraper
from rag import RAGTrends
from tweet import generate_tweet

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

st.set_page_config(
    layout="centered",
    page_title="Keep Up With The Trends",
    page_icon="📋",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "This app is a part of a tutorial by [MLWH](https://www.extremelycoolapp.com) to perform RAG for news articles scraping.",
    },
)

st.title("Keep Up With The Trends")
st.subheader(":red[All the trending topics summarized in one place.]")

@st.cache_resource
def get_models():
    llm = ChatOllama(model="llama3.1", temperature=0.2, keep_alive="2m", repeat_penalty=1.03)
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings_model

llm, embeddings_model = get_models()

trend_realtime_scraper = TrendScraper("realtime")
trend_daily_scraper = TrendScraper("daily")
rag_chain = RAGTrends(llm_ollama_model=llm, hf_embeddings_model=embeddings_model)

gg_daily_data_dict = trend_realtime_scraper.fetch_and_parse_xml()
gg_realtime_data_dict = trend_realtime_scraper.fetch_and_parse_xml()

daily_trend_title_list = [item['title'] for item in gg_daily_data_dict['rss']['channel']['item']]
realtime_trend_title_list = [item['title'] for item in gg_realtime_data_dict['rss']['channel']['item']]

    
st.divider()
col1, col2 = st.columns(2)
with col1:
    top_k_daily = st.slider("Top k daily trends", min_value=1, max_value=len(daily_trend_title_list), value=3)
with col2:
    top_k_realtime = st.slider("Top k realtime trends", min_value=1, max_value=len(realtime_trend_title_list), value=3)

@st.cache_data
def scrape_daily_trends(_daily_data_dict, daily_trend_title_list):
    print("Running scrape_daily_trends for real!")
    results_dfs = trend_daily_scraper.run(_daily_data_dict)
    return results_dfs

@st.cache_data
def scrape_realtime_trends(_realtime_data_dict, realtime_trend_title_list):
    print("Running scrape_realtime_trends for real!")
    results_dfs = trend_realtime_scraper.run(_realtime_data_dict)
    return results_dfs

google_daily_df, ddg_daily_df = scrape_daily_trends(gg_daily_data_dict, daily_trend_title_list)
google_realtime_df, ddg_realtime_df = scrape_realtime_trends(gg_realtime_data_dict, realtime_trend_title_list)


def extract_domain(url):
    """
    Extracts the domain from a given URL.

    Parameters:
        url (str): The URL from which to extract the domain.

    Returns:
        str or None: The extracted domain if found, otherwise None.
    """
    pattern = re.compile(r"https?://(?:www\.)?([a-zA-Z0-9-]+)\.([a-zA-Z]{2,})")
    match = pattern.search(url)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None

@st.cache_data
def run_rag_cached(_rag_chain_instance, _gg_df, _ddg_df, kws):
    print("Running run_rag_cached for real!")
    results = _rag_chain_instance.run_rag(_gg_df, _ddg_df)
    return results

tab1, tab2 = st.tabs(["Daily Trends", "Realtime Trends"])

with tab1:
    with st.expander("Show scraping results", expanded=False):
        st.write("Google search daily trends results")
        st.write(google_daily_df.iloc[:top_k_daily,:])
        st.write("")
        st.write("Duck-Duck-Go News results for Google daily trends")
        st.write(ddg_daily_df.iloc[:top_k_daily*3,:])
    rag_daily_results = run_rag_cached(rag_chain, google_daily_df.iloc[:top_k_daily,:], ddg_daily_df.iloc[:top_k_daily*3,:], google_daily_df.loc[:top_k_daily,"trend_kws"].to_list())
    st.toast('Daily trends results ready!', icon='🔥')
    #rag_daily_results = rag_chain.run_rag(google_daily_df.iloc[:top_k_daily,:], ddg_daily_df.iloc[:top_k_daily,:])
    for i in range(len(rag_daily_results["Trend_kws"])):
        with st.expander(f"**{rag_daily_results['Title'][i]}**", expanded=False):
            st.write(rag_daily_results["Summary"][i])
            st.divider()
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                st.write(f"🚀 Traffic: {google_daily_df['traffic'].iloc[i]} Google Searches")
            with col2:
                st.write(f"📅 Date: {pd.Timestamp(google_daily_df['pubDate'].iloc[i]).strftime('%a %d %b %Y, %I:%M%p')}")
            
            domains_list = [extract_domain(link) for link in google_daily_df["url"].iloc[i]]
            st.markdown(f"🌐 Refrences: " + ", ".join([f"[{domains_list[j]}]({google_daily_df['url'].iloc[i][j]})" for j in range(len(google_daily_df['url'].iloc[i][:3]))]))

            col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
            with col1:
                sent_menu_daily = st.selectbox(f"Sentiment", ["positive", "neutral", "negative"], key=f"sentiment_daily_{i}")
            with col2:
                tone_menu_daily = st.selectbox(f"Tone", ["Professional", "Friendly", "Sarcastic", "Aggressive", "Skeptical", "Shakespearean"], key=f"tone_daily_{i}")
            with col3:
                gen_twt_daily = st.button(f"Generate Tweet", key=f"tweet_btn_daily_{i}")
                
            if gen_twt_daily:
                resp = generate_tweet(llm, rag_daily_results["Title"][i], rag_daily_results["Summary"][i], sent_menu_daily, tone_menu_daily)
                st.write(f"#️⃣Tweet:\n {resp.content}")
    
with tab2:
    with st.expander("Show scraping results", expanded=False):
        st.write("Google search realtime trends results")
        st.write(google_realtime_df.iloc[:top_k_realtime,:])
        st.write("")
        st.write("Duck-Duck-Go News results for Google realtime trends")
        st.write(ddg_realtime_df.iloc[:top_k_realtime*3,:])
    rag_realtime_results = run_rag_cached(rag_chain, google_realtime_df.iloc[:top_k_realtime,:], ddg_realtime_df.iloc[:top_k_realtime*3,:], google_realtime_df.trend_kws.to_list())
    st.toast('Rrealtime trends results ready!', icon='🔥')
    #rag_daily_results = rag_chain.run_rag(google_daily_df.iloc[:top_k_daily,:], ddg_daily_df.iloc[:top_k_daily,:])
    for i in range(len(rag_realtime_results["Trend_kws"])):
        with st.expander(f"**{rag_realtime_results['Title'][i]}**", expanded=False):
            st.write(rag_realtime_results["Summary"][i])
            st.divider()
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                st.write(f"🚀 Traffic: {google_realtime_df['traffic'].iloc[i]} Google Searches")
            with col2:
                st.write(f"📅 Date: {pd.Timestamp(google_realtime_df['pubDate'].iloc[i]).strftime('%a %d %b %Y, %I:%M%p')}")
            
            domains_list = [extract_domain(link) for link in google_realtime_df["url"].iloc[i]]
            st.markdown(f"🌐 Refrences: " + ", ".join([f"[{domains_list[j]}]({google_realtime_df['url'].iloc[i][j]})" for j in range(len(google_realtime_df['url'].iloc[i][:3]))]))

            col1, col2, col3 = st.columns(3, vertical_alignment="bottom")
            with col1:
                sent_menu_realtime = st.selectbox(f"Sentiment", ["positive", "neutral", "negative"], key=f"sentiment_realtime_{i}")
            with col2:
                tone_menu_realtime = st.selectbox(f"Tone", ["Professional", "Friendly", "Sarcastic", "Aggressive", "Skeptical", "Shakespearean"], key=f"tone_realtime_{i}")
            with col3:
                gen_twt_realtime = st.button(f"Generate Tweet", key=f"tweet_btn_realtime_{i}")
                
            if gen_twt_realtime:
                resp = generate_tweet(llm, rag_realtime_results["Title"][i], rag_realtime_results["Summary"][i], sent_menu_realtime, tone_menu_realtime)
                st.write(f"#️⃣Tweet:\n {resp.content}")