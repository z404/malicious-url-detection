import pandas as pd
from xgboost.sklearn import XGBClassifier
import pickle
import re

is_running_status = False

def preprocess_dataset(df):
    df['-cnt'] = df['url'].str.count('-')
    df['?cnt'] = df['url'].str.count('\?')
    df['%cnt'] = df['url'].str.count('\%')
    df['@cnt'] = df['url'].str.count('@')
    df['=cnt'] = df['url'].str.count('=')
    df['.cnt'] = df['url'].str.count('.')
    # Counting http and https occurrences
    df['httpcnt'] = df['url'].str.count('http')
    # df['httpscnt'] = df['url'].str.count('https')
    # df['wwwcnt'] = df['url'].str.count('www')

    df['digitcnt'] = df['url'].str.count('\d')
    df['lettercnt'] = df['url'].str.count('[A-Za-z]')
    df['dircnt'] = df['url'].str.count('/') - 2

    # Obtaining length of the url
    df["length_of_url"] = df["url"].apply(lambda url: len(url))
    # Obtaining length of path
    df["length_of_path"] = df["url"].apply(lambda url: len("/".join(url.split("/")[3:])))
    # Obtaining length of the hostname
    df["length_of_hostname"] = df["url"].apply(lambda url: len(url.split("/")[2]))
    # Obtaining length of the first directory
    df["length_of_fd"] = df["url"].apply(lambda url: len(url.split("/")[3]) if len(url.split("/")) > 3 else 0)
    # Obtaining length of the top level directory
    df["length_of_td"] = df["url"].apply(lambda url: len(url.split("/")[2].split(".")[-1]))

    def is_domain_an_ip(url):
        return url.split("/")[2].split(".")[-1].isdigit()

    def is_domain_a_shortened_link(url):
        match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                        'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                        'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                        'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                        'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                        'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                        'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                        'tr\.im|link\.zip\.net',
                        url)
        if match:
            return 1
        else:
            return 0

    df["is_ip"] = df["url"].apply(is_domain_an_ip)
    df["is_shortened_link"] = df["url"].apply(is_domain_a_shortened_link)

    return df

def run():
    global is_running_status
    is_running_status = True
    
    df = pd.read_csv("urldata.csv")
    df = preprocess_dataset(df)

    #Independent Variables
    x = df[['length_of_hostname',
        'length_of_path', 'length_of_fd', 'length_of_td', '-cnt', '@cnt', '?cnt',
        '%cnt', '.cnt', '=cnt', 'httpcnt', 'digitcnt', #'httpscnt', 'wwwcnt',
        'lettercnt', 'dircnt', 'is_ip']]

    #Dependent Variable
    y = df['result']

    model = XGBClassifier().fit(x, y)
    pickle.dump(model, open("model.pkl", "wb"))

    is_running_status = False

def is_running():
    return is_running_status