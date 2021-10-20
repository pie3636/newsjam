from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import re
import json
import datetime

def actu_scraper(url):
    '''
    takes a given string URL and scrapes all relevant information (for us)
    returns a dictionary with all relevant information to be converted into JSON
    '''
    #initialising all the webdriver stuff
    DRIVER_PATH = 'C:/bin/chromedriver.exe'
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options, executable_path=DRIVER_PATH)
    driver.get(url)

    #bypasses the cookie popup and grabs the article
    button = driver.find_element_by_css_selector("#didomi-notice-agree-button")
    button.click()
    article = driver.find_elements_by_xpath("/html/body/div[2]/main/div/div[1]/div[1]")
    article_text = []

    #converts all dates to the format dd/mm/yyyy, and grabs the title, author, summary, etc.
    date = driver.find_elements_by_xpath("/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]/div/div[1]/span/time")
    date = re.sub(r'\sà.*',"",date[0].text)
    date = datetime.datetime.strptime(date,'%d %b %y').strftime("%d/%m/%Y")
    title = driver.find_elements_by_xpath("/html/body/div[2]/main/div/div[1]/div[1]/article/div[1]/h1")
    title = title[0].text
    author = driver.find_elements_by_xpath("/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]/div/div[1]/strong/a")
    author = author[0].text
    summary = driver.find_elements_by_xpath("/html/body/div[2]/main/div/div[1]/div[1]/article/div[1]/p")
    summary = summary[0].text

    #article text cleanup, automatically removes all image captions, extraneous information at the start and end of every article, etc.
    for x in range(len(article)):
        article_text.append(article[x].text.replace("\n"," "))
        article_text[x] = re.sub(r'Par.*\d{1,2}:\d{1,2}|Cet article vous a été.*|\(©.*?\)|<a.*?a>',"",article_text[x])
    driver.close()
    json_dict = {'title':title,'date':date,'url':url,'author':author,'topic':'societe','summary':summary,'text':str(article_text[0])}
    return json_dict
	
def json_converter(url, file):
    '''
    takes a list of URLs and a desired filename
    scrapes every URL in the list using the actu_scraper function
    then places them in a json file named after the parameter 'file'
    '''
    for x in range(len(url)):
        with open(file, "a", encoding='utf-16') as outfile:
            outfile.write(json.dumps(actu_scraper(url[x]),indent=4,ensure_ascii=False))
            outfile.write(',\n')
			
def autoscraper(url):
    '''
    automatically grabs every url on a given index page for actu.fr
    (except for the first 3)
    returns a list of URLs as strings
    '''
    DRIVER_PATH = 'C:/bin/chromedriver.exe'
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options, executable_path=DRIVER_PATH)
    driver.get(url)
    element2 = driver.find_element_by_xpath('/html/body/div[2]/main/div/div[2]/div/div/ul').find_elements_by_tag_name('a')
    test_element = []
    for element in element2:
        test_element.append(element.get_attribute("href"))
    return test_element