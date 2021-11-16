from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import re
import json
import datetime

def actu_scraper(url):
    '''
    takes a given string URL and scrapes all relevant information (for us)
    returns a dictionary with all relevant information to be converted into JSON
    '''
    #initialising all the webdriver stuff
    ser = Service('C:/bin/chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options, service=ser)
    driver.get(url)

    #bypasses the cookie popup and grabs the article
    button = driver.find_element(By.CSS_SELECTOR, "#didomi-notice-agree-button")
    button.click()
    article = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div/div[1]/div[1]")
    article_text = []

    #converts all dates to the format dd/mm/yyyy, and grabs the title, author, summary, etc.
    date = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]/div/div[1]/span/time")
    date = re.sub(r'\sà.*',"",date[0].text)
    date = datetime.datetime.strptime(date,'%d %b %y').strftime("%d/%m/%Y")
    title = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div/div[1]/div[1]/article/div[1]/h1")
    title = title[0].text
    author = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div/div[1]/div[1]/article/div[2]/div/div[1]/strong/a")
    author = author[0].text
    summary = driver.find_elements(By.XPATH, "/html/body/div[2]/main/div/div[1]/div[1]/article/div[1]/p")
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
    if url == []:
        pass
    else:
        for x in range(len(url)):
            with open(file, "a", encoding='utf-16') as outfile:
                outfile.write(json.dumps(actu_scraper(url[x]),indent=4,ensure_ascii=False))
                outfile.write(',\n')
			
def autoscraper(url, file):
    '''
    automatically grabs every url on a given index page for actu.fr
    (except for the first 3)
    returns a list of URLs as strings and appends URLs to a master list for checks
    '''
    outfile = open(file, "r", encoding='utf-16')
    temp_list = outfile.readlines() #read lines for master list comparisons
    outfile.close()
    outfile = open(file, "a", encoding='utf-16') #open in append mode
    ser = Service('C:/bin/chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(options=chrome_options, service=ser)
    driver.get(url)
    element2 = driver.find_element(By.XPATH, '/html/body/div[2]/main/div/div[2]/div/div/ul').find_elements(By.TAG_NAME, 'a')
    test_element = []
    for element in element2:
        if element.get_attribute('href')+'\n' not in temp_list: #duplicate check
            test_element.append(element.get_attribute("href"))
            outfile.write(element.get_attribute("href")+'\n')
        else:
            continue
    return test_element

def page_skipper(url, *page_num, file):
    '''
    simply appends the given page numbers to the actu url to go to the next page
    since each call returns a lsit of lists, it also flattens the list to be used as input for the json_converter function
    '''
    url_list = []
    for x in range(*page_num):
        url_list.append(autoscraper(url+'/page/{0}'.format(x), file)) #runs autoscraper, appending each url output as a list in a list
    url_list = [x for sublist in url_list for x in sublist] #flattens the list of lists to just a list
    print("new articles added:", len(url_list)) #simple check to see if there was any point in running the scraper
    return url_list

json_converter(list(page_skipper('https://actu.fr/societe/coronavirus', 0, 20, file='masterlist_actu.txt')), 'test.json') #sample function call
#note that you can type just one number, or you can include a range of pages you would like to include
