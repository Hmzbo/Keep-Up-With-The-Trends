import pandas as pd
import requests
import xmltodict
from duckduckgo_search import DDGS
import streamlit as st


class TrendScraper():
    """
    Scrape Google Trends daily(past 48 hours) and realtime(past 4 hours) data.
    """
    def __init__(self, trend_type:str="realtime"):
        """
        Initializes a new instance of the TrendScraper class.

        Args:
            trend_type (str, optional): The type of trends to fetch. Defaults to "realtime".
                Must be either "daily" or "realtime".

        Raises:
            ValueError: If the provided trend_type is not "daily" or "realtime".

        """
        self.rss_xml_url_daily = 'https://trends.google.fr/trends/trendingsearches/daily/rss?geo=US'
        self.rss_xml_url_realtime = 'https://trends.google.com/trending/rss?geo=US'
        self.domains_skip_list = ["msn.com", "nytimes.com"]
        if trend_type == "daily":
            self.rss_xml_url = self.rss_xml_url_daily
        elif trend_type == "realtime":
            self.rss_xml_url = self.rss_xml_url_realtime
        else:
            raise ValueError("trend_type must be either 'daily' or 'realtime'.")
        pass


    def fetch_and_parse_xml(self):
        """
        Fetches XML data from a given URL and parses it into a dictionary.

        Returns:
            dict: A dictionary representation of the parsed XML data.

        Raises:
            requests.exceptions.RequestException: If an error occurs while fetching the XML data.
            xmltodict.expat.ExpatError: If an error occurs while parsing the XML data.
        """
        try:
            # Fetch the XML data
            response = requests.get(self.rss_xml_url)
            response.raise_for_status()  # Raise an error for bad status codes
            xml_data = response.content

            # Parse the XML data and convert it to a dictionary
            data_dict = xmltodict.parse(xml_data)
            return data_dict

        except requests.exceptions.RequestException as e:
            print(f"Error fetching the XML data: {e}")
        except xmltodict.expat.ExpatError as e:
            print(f"Error parsing the XML data: {e}")

    def create_google_dataframes(self, trends_dict):
        """
        Creates a pandas DataFrame from a dictionary containing Google Trends data.

        Args:
            trends_dict (dict): A dictionary containing the Google Trends data. It should have the following structure:
                - 'rss' (dict): A dictionary containing the RSS feed data.
                    - 'channel' (dict): A dictionary containing the channel data.
                        - 'item' (list): A list of dictionaries containing the individual trend data.
                            - 'title' (str): The title of the trend.
                            - 'ht:approx_traffic' (int): The approximate traffic for the trend.
                            - 'pubDate' (str): The publication date of the trend.
                            - 'ht:news_item' (list or dict): A list or dictionary containing the news items for the trend.

            Note: The 'ht:news_item' field can be either a list or a dictionary. If it is a list, each item in the list is a dictionary with the following keys:
                - 'ht:news_item_url' (str): The URL of the news item.
                - 'ht:news_item_title' (str): The title of the news item.

                If it is a dictionary, it has the following keys:
                - 'ht:news_item_url' (str): The URL of the news item.
                - 'ht:news_item_title' (str): The title of the news item.

        Returns:
            google_trends_df (pandas.DataFrame): A DataFrame containing the Google Trends data. It has the following columns:
                - 'trend_kws' (list): The titles of the trends.
                - 'traffic' (list): The approximate traffic for the trends.
                - 'pubDate' (list): The publication dates of the trends.
                - 'url' (list): The URLs of the news items for each trend.
                - 'title' (list): The titles of the news items for each trend.
                - Note: The 'url' and 'title' columns may contain multiple values for each trend if there are multiple news items.

        """
        google_trends_dict = {"trend_kws":[], "traffic":[], "pubDate":[], "url":[], "title":[]}
        for trend in trends_dict['rss']['channel']['item']:
            google_trends_dict['trend_kws'].append(trend['title'])
            google_trends_dict['traffic'].append(trend['ht:approx_traffic'])
            google_trends_dict['pubDate'].append(trend['pubDate'])
            if isinstance(trend['ht:news_item'], list):
                google_trends_dict['url'].append([news_item['ht:news_item_url'] for news_item in trend['ht:news_item']])
                google_trends_dict['title'].append([news_item['ht:news_item_title'] for news_item in trend['ht:news_item']])
            else:
                google_trends_dict['url'].append([trend['ht:news_item']['ht:news_item_url']])
                google_trends_dict['title'].append([trend['ht:news_item']['ht:news_item_title']])

        google_trends_df = pd.DataFrame(google_trends_dict)
        google_trends_df["title"] = google_trends_df["title"].map(lambda links: [link.replace("&#39;", "'") for link in links])

        for i in range(google_trends_df.shape[0]):
            for j, url in enumerate(google_trends_df.loc[i,"url"]):
                if any(domain in url for domain in self.domains_skip_list):
                    _ = google_trends_df.loc[i,"url"].pop(j)
                    _ = google_trends_df.loc[i,"title"].pop(j)
        return google_trends_df
    

    def create_ddg_dataframe(self, google_trends_df):
        """
        Creates a DataFrame of DDG news results for each trend keyword in the given Google trends DataFrame.

        Args:
            google_trends_df (pandas.DataFrame): The Google trends DataFrame containing the trend keywords.

        Returns:
            pandas.DataFrame: The DataFrame of DDG news results, with each row representing a news result.
        """

        trends_news = []
        for trend_kw in google_trends_df.trend_kws.to_list():
            results = DDGS().news(keywords=trend_kw, region="wt-wt", safesearch="moderate", max_results=15)
            filtered_results = [res for res in results if not any(domain in res['url'] for domain in self.domains_skip_list)]
            filtered_results = list(map(lambda d: {'trend_kws':trend_kw, **d}, filtered_results[:3]))
            trends_news.extend(filtered_results)
        
        trends_ddg_news_df = pd.DataFrame(trends_news)
        return trends_ddg_news_df


    def run(self, data_dict):

        trends_dict = self.fetch_and_parse_xml()
        google_trends_df = self.create_google_dataframes(trends_dict)
        trends_ddg_news_df = self.create_ddg_dataframe(google_trends_df)

        for i, trend in enumerate(google_trends_df.trend_kws):
            url_list = trends_ddg_news_df[trends_ddg_news_df["trend_kws"]==trend]['url'].to_list()
            title_list = trends_ddg_news_df[trends_ddg_news_df["trend_kws"]==trend]['title'].to_list()
            google_trends_df.loc[i, "url"].extend(url_list)
            google_trends_df.loc[i, "title"].extend(title_list)

        return google_trends_df, trends_ddg_news_df