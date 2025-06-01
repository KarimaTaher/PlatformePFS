# scraping.py
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import datetime
import requests

month_map = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

def get_us_inflation_rate():
    options = Options()
    options.add_argument("user-agent=Mozilla/5.0")
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    driver.get("https://tradingeconomics.com/united-states/inflation-cpi")
    try:
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "calendar")))
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", id="calendar")
        if not table:
            return None
        tbody = table.find("tbody")
        rows = tbody.find_all("tr")
        current_month = datetime.now().strftime("%b")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                reference_month = cols[3].get_text(strip=True)
                if reference_month == current_month:
                    return cols[4].get_text(strip=True).replace('%', '').strip()
        return None
    finally:
        driver.quit()

def get_us_petroleum_production():
    url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS2&f=M'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        current_year = str(datetime.now().year)
        current_month_name = month_map[datetime.now().month]
        tbody = soup.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                if tds and tds[0].get_text(strip=True) == current_year:
                    row_data = [td.get_text(strip=True).replace('\xa0', '') for td in tds]
                    columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    data_dict = dict(zip(columns, row_data))
                    value = data_dict.get(current_month_name, None)
                    if value:
                        return value.replace(',', '.')
    return None

def get_us_petroleum_export():
    url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=pet&s=mcrexus1&f=m'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        current_year = str(datetime.now().year)
        current_month_name = month_map[datetime.now().month]
        tbody = soup.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                year_cell = tds[0].get_text(strip=True).replace('\xa0', '').replace(' ', '')
                if year_cell == current_year:
                    columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    # Remplacer les virgules par des points et nettoyer les valeurs
                    row_data = [td.get_text(strip=True).replace('\xa0', '').replace(',', '.') for td in tds]
                    if len(row_data) < len(columns):
                        row_data += [''] * (len(columns) - len(row_data))
                    data_dict = dict(zip(columns, row_data))
                    # Obtenir la valeur du mois en tant que nombre à virgule flottante
                    value_str = data_dict.get(current_month_name, None)
                    if value_str:
                        try:
                            # Convertir en nombre flottant (milliers de barils par jour)
                            value = float(value_str)
                            return value
                        except ValueError:
                            pass
    return None

def get_us_petroleum_import():
    url = 'https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MTTIMUS1&f=M'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        current_year = str(datetime.now().year)
        current_month_name = month_map[datetime.now().month]
        tbody = soup.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                tds = tr.find_all('td')
                year_cell = tds[0].get_text(strip=True).replace('\xa0', '').replace(' ', '')
                if year_cell == current_year:
                    columns = ['Year', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    # Remplacer la virgule par un point
                    row_data = [td.get_text(strip=True).replace('\xa0', '').replace(',', '.') for td in tds]
                    if len(row_data) < len(columns):
                        row_data += [''] * (len(columns) - len(row_data))
                    data_dict = dict(zip(columns, row_data))
                    # Obtenir la valeur du mois en nombre flottant
                    value_str = data_dict.get(current_month_name, None)
                    if value_str:
                        try:
                            value = float(value_str)
                            return value
                        except ValueError:
                            pass
    return None

