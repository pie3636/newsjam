# -*- coding: utf-8 -*-
"""scraper_functions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1iDJxmAFphLAcVAO4cpANIWujfud5B1c2
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from itertools import groupby
from selenium.webdriver.chrome.options import Options
import re
import sys
import time

sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver') #make sure the webdriver is in the right spot

def actu_scraper(url):
    '''
    takes a given URL, grabs the article text
    returns the text as a string
    '''
    #webdriver initialization stuff
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver',options=options)
    driver.get(url)

    #bypasses the cookie popup and grabs the article
    try:
      button = driver.find_element(By.CSS_SELECTOR, "#didomi-notice-agree-button")
      button.click()
    except:
      article = driver.find_elements(By.XPATH, '/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]')
    time.sleep(2)
    article = driver.find_elements(By.XPATH, '/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]')
    article_text = []
    
    #article text cleanup, automatically removes all image captions, extraneous information at the start and end of every article, etc.
    for x in range(len(article)):
        article_text.append(article[x].text.replace("\n"," "))
        article_text[x] = re.sub(r'Par.*\d{1,2}:\d{1,2}|Cet article vous a été.*|\(©.*?\)|<a.*?a>',"",article_text[x])
    driver.close()
    return article_text[x], url

def actu_autoscraper(url, url_amount=5):
    '''
    grab the given amount of URLs from an actu.fr list of articles and return them as a list
    '''
    url_list = []
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome('chromedriver',options=options)
    driver.get(url)
    element = driver.find_element(By.XPATH, '/html/body/div[2]/main/div/div[2]/div/div/ul').find_elements(By.TAG_NAME, 'a')
    for x in range(url_amount):
      url_list.append(element[x].get_attribute("href"))
    return url_list
