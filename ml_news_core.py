import warnings
import re
from newspaper import Article

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from duckduckgo_search import ddg
import joblib
from googlesearch import search
from urllib.parse import urlparse
import operator
from itertools import islice
import pandas as pd

warnings.filterwarnings("ignore")

df = pd.read_excel("output.xls", header=None)

match = df[0].to_list()


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def extractor(url):
    """
    Extractor function that gets the article body from the URL
    Args:
        url: URL of the News Article/Source
    Returns:
        article: Raw Article Body
        article_title: Title of the Article that has been extracted
    """

    article = Article(url)

    article.download()
    article.parse()
    

    #Get the article title and convert them to lower-case
    article_title = article.title
    # print("--q1231")
    # print(article_title)
    # print(article)
    article = article.text.lower()
    article = [article]
    return (article, article_title)


def text_area_extractor(text):
    """
    Textbox extractor function to preprocess and extract text
    Args:
        text: Raw Extracted Text from the News Article
    Returns:
        text: Preprocessed and clean text ready for analysis
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub("(\\r|\r|\n)\\n$", " ", text)
    text = [text]
    return text

def duckduckgo_search(title):
    """
    Function to perform a Google Search with the specified title and URL
    Args:
        title: Title of the Article
        url: URL of the specified article
    Returns:
        search_urls: Similar News Articles found over the Web
        source_sites: Hostname of the Articles founder over the Web
    """
    # target = url
    # domain = urlparse(target).hostname
    # print(title)
    title = title.split(" – Balanced News Summary")[0]
    # print(title)
    search_urls = []
    source_sites = []
    results = ddg(title, region='wt-wt', safesearch='Moderate', time='y', max_results=10)
    for result in results:
        if any(x in result["href"] for x in match):
        #if "https://balancednewssummary.com/" not in result["href"]:
            source_sites.append(result["title"])
            search_urls.append(result["href"])
    # for i in search(title, tld = "com", num = 10, start = 1, stop = 6):
    #     if "youtube" not in i and domain not in i:
    #         source_sites.append(urlparse(i).hostname)
    #         search_urls.append(i)
    return search_urls, source_sites

def similarity(url_list, article):
    """
    Function to check the similarity of the News Article through Cosine Similarity
    Args:
        url_list: List of the URLs similar to the news article
        article: Preprocessed article which would be vectorized
    Returns:
        cosine_cleaned: Cosine Similarity Scores of each URL passed
        average_score: Average value of the cosine similarity scores fetched
    """
    article = article
    sim_tfv = TfidfVectorizer(stop_words ="english")
    sim_transform1 = sim_tfv.fit_transform(article)
    cosine = []
    cosine_cleaned = []
    cosine_average = 0
    count = 0
    article_titles = []
    for i in url_list:
        try: 
            test_article, test_title = extractor(i)
            test_article = [test_article]
            sim_transform2 = sim_tfv.transform(test_article[0])
            score = cosine_similarity(sim_transform1, sim_transform2)
            cosine.append(score*100)
            article_titles.append(test_title)
            count+=1
        except Exception as e:
            print(e) #naked except for handling failed downloads
    for i in cosine:
        x = str(i).replace('[','').replace(']','')
        cosine_cleaned.append(float(x))

    for i in cosine:
        if i !=0:
            cosine_average = cosine_average + i
        else:
            count-=1

    average_score = cosine_average/count
    average_score = str(average_score).replace('[','').replace(']','')
    average_score = float(average_score)
    # return cosine_cleaned, average_score
    return cosine_cleaned,article_titles

def handlelink(article_link):
    """
    Classification function to take the article link and predict the similar news articles
    Args:
        article_link: URL of the article
    Returns:
        pred: Predicted news sources from the machine learning model
        article_title: Title of the Article
        article: Article fetched from the URL 
        url: URL of the article
    """

    job_pac = joblib.load('models/pac.pkl')
    job_vec = joblib.load('models/tfv.pkl')
    url = (article_link)
    article, article_title = extractor(article_link)
    # print("_------------")
    # print(article_link)
    # print(article)
    # print(article_title)
    pred = job_pac.predict(job_vec.transform(article))
    return pred, article_title, article, url


def similarNews(url):
    """
    Driver function to return a dictionary with all the similar news and their similarity score
    Args:
        url: URL of the article
    Returns:
        dictionary: Dictionary containing all the similar news articles and their similarity score
    """
    prediction, article_title, article, url = handlelink(article_link=url)
    url_list, article_titles = duckduckgo_search(article_title)
    # similarity_score, avgScore = similarity(url_list, article)
    similarity_score, article_titles_old = similarity(url_list, article)
    dictionary = dict(zip(url_list, similarity_score))
    url_title_dict = dict(zip(url_list, article_titles))
    
    sorted_d = dict( sorted(dictionary.items(), key=lambda item:item[1],reverse=True))
    print(sorted_d)
    for key in sorted_d.keys():
        for index, row in df.iterrows():
            if row[0] in key:
                sorted_d[key] = row[1]

    l=[]
    c=[]
    r=[]
    
    for key, value in sorted_d.items():
        if value in ["L", "CL"]:
            l.append({key:url_title_dict[key]})
        elif value in ["C"]:
            c.append({key:url_title_dict[key]})
        elif value in ["R", "CR"]:
            r.append({key:url_title_dict[key]})

    lun = [len(l), len(c), len(r)]
    final = []

    for i in range(max(lun)):
        try:
            final.append(l[i])
        except:
            pass
        try:
            final.append(c[i])
        except:
            pass
        try:
            final.append(r[i])
        except:
            pass


    # json_response = [{key:url_title_dict[key], "rating":value} for key, value in sorted_d.items()]

    # for item in json_response:
    #     for index, row in df.iterrows():
    #         if row[0] in item.key:
    #             print(row[0], row[1])
    # print(sorted_d)
    # # url_list = [{key:n_items[key]} for key in n_items]
    # url_list = [{key:article_titles[index]} for index,key in enumerate(sorted_d)]
    # dictionary = [{item,similarity_score[index]} for index,item in enumerate(url_list)]
    # print(dictionary)
    # print(url_title_dict)
    # print(json_response)
    return final[0:3]




if __name__ == "__main___":
    pass
    

