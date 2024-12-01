from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd

URL = 'https://rating.hh.ru/history/rating2023/summary?tab=giant"'

XPATHS = [
    '//*[@id="root"]/div/div/div/div[1]/div/div[7]/div[1]/div[1]',
    '//*[@id="root"]/div/div/div/div[1]/div/div[7]/div[1]/div[2]',
    '//*[@id="root"]/div/div/div/div[1]/div/div[7]/div[1]/div[3]',
    '//*[@id="root"]/div/div/div/div[1]/div/div[7]/div[1]/div[4]'
]

NAMETAGS = ['Крупнейшие', 'Крупные', 'Средние', 'Небольшие']

service = Service(executable_path=r'parsers\chromedriver-win64\chromedriver.exe')
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=service, options=options)
driver.maximize_window()
driver.get(URL)

dataframes = {} 
for name, xpath in zip(NAMETAGS, XPATHS): 

    element = driver.find_element(By.XPATH, xpath)
    ActionChains(driver).move_to_element(element).click().perform()
    
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CLASS_NAME, '_176_F2jG6H'))
    )

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table', class_='_176_F2jG6H')

    data = []
    rows = table.find('tbody').find_all('tr')
    for row in rows:
        columns = row.find_all('td')
        
        row_data = [col.get_text(strip=True) for col in columns]

        data.append(row_data)

    headers = [header.get_text(strip=True) for header in table.find_all("th")]
    df = pd.DataFrame(data, columns=headers)
    dataframes[name] = df
    
for name, df in dataframes.items():
    df.to_csv(rf'parsed\{name}_HH_rating_2023.csv')
