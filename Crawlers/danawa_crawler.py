# -*- coding: utf-8 -*-

# 1. 라이브러리 임포트
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from datetime import datetime
from datetime import timedelta
from pytz import timezone

import csv
import os
import os.path
import shutil
import traceback
import re
from multiprocessing import Pool
from github import Github

# 2. 환경 설정
IS_TEST = False  # 테스트 시 True
PROCESS_COUNT = 2 

GITHUB_TOKEN_KEY = 'MY_GITHUB_TOKEN'
GITHUB_REPOSITORY_NAME = 'sammy310/Danawa-Crawler'

CRAWLING_DATA_CSV_FILE = 'CrawlingCategory.csv'
if IS_TEST:
    CRAWLING_DATA_CSV_FILE = 'CrawlingCategory_test.csv'

DATA_PATH = '../dataset/price_history'
DATA_REFRESH_PATH = f'{DATA_PATH}/Last_Data'
SPEC_DATA_PATH = '../dataset/specs'

TIMEZONE = 'Asia/Seoul'
CHROMEDRIVER_PATH = 'chromedriver'

DATA_DIVIDER = '---'
DATA_REMARK = '//'
DATA_ROW_DIVIDER = '_'
DATA_PRODUCT_DIVIDER = '|'

STR_NAME = 'name'
STR_URL = 'url'
STR_CRAWLING_PAGE_SIZE = 'crawlingPageSize'

# 3. 크롤러 클래스 시작
class DanawaCrawler:
    def __init__(self):
        self.errorList = list()
        self.crawlingCategory = list()
        
        # 폴더 생성
        self.CheckAndCreateDirs()

        # 크롤링 리스트 읽기
        try:
            with open(CRAWLING_DATA_CSV_FILE, 'r', newline='') as file:    
                for crawlingValues in csv.reader(file, skipinitialspace=True):  
                    if not crawlingValues[0].startswith(DATA_REMARK):
                        self.crawlingCategory.append({STR_NAME: crawlingValues[0], 
                                                      STR_URL: crawlingValues[1],  
                                                      STR_CRAWLING_PAGE_SIZE: int(crawlingValues[2])})
        except FileNotFoundError:
            print(f"[Error] {CRAWLING_DATA_CSV_FILE} 파일을 찾을 수 없습니다.")

    def CheckAndCreateDirs(self):
        if not os.path.exists('../dataset'): os.makedirs('../dataset')
        if not os.path.exists(DATA_PATH): os.makedirs(DATA_PATH)
        if not os.path.exists(SPEC_DATA_PATH): os.makedirs(SPEC_DATA_PATH)

    def StartCrawling(self):
        # 크롬 옵션 설정 (헤드리스 등)
        self.chrome_option = Options()
        self.chrome_option.add_argument('--headless') 
        self.chrome_option.add_argument('--window-size=1920x1080')
        self.chrome_option.add_argument('--start-maximized')
        self.chrome_option.add_argument('--disable-gpu')
        self.chrome_option.add_argument('lang=ko=KR')
        custom_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
        self.chrome_option.add_argument(f'user-agent={custom_user_agent}')
        self.chrome_option.add_argument('--no-sandbox')
        self.chrome_option.add_argument('--disable-dev-shm-usage')

        if __name__ == '__main__':
            pool = Pool(processes=PROCESS_COUNT)
            pool.map(self.CrawlingCategory, self.crawlingCategory)
            pool.close()
            pool.join()

    def CrawlingCategory(self, categoryValue):
        crawlingName = categoryValue[STR_NAME]
        crawlingURL = categoryValue[STR_URL]
        crawlingSize = categoryValue[STR_CRAWLING_PAGE_SIZE]

        print('Crawling Start : ' + crawlingName)

        # 파일 1: 가격 데이터 (기존 로직 유지)
        crawlingFile = open(f'{crawlingName}.csv', 'w', newline='', encoding='utf8')
        crawlingData_csvWriter = csv.writer(crawlingFile)
        crawlingData_csvWriter.writerow([self.GetCurrentDate().strftime('%Y-%m-%d %H:%M:%S')])
        
        # 파일 2: 스펙 데이터 (ID, Name, Spec_Details 만 저장)
        specTempFile = open(f'{crawlingName}_Spec_Temp.csv', 'w', newline='', encoding='utf8')
        spec_csvWriter = csv.writer(specTempFile)
        spec_csvWriter.writerow(['ID', 'Name', 'Spec_Details'])

        try:
            browser = webdriver.Chrome(options=self.chrome_option)
            browser.implicitly_wait(5)
            browser.get(crawlingURL)

            # 90개씩 보기 클릭
            try: browser.find_element(By.XPATH, '//option[@value="90"]').click()
            except: pass
        
            wait = WebDriverWait(browser, 10)
            try: wait.until(EC.invisibility_of_element((By.CLASS_NAME, 'product_list_cover')))
            except: pass
            
            # 페이지 반복 (BEST + 1페이지 ~ N페이지)
            for i in range(-1, crawlingSize):
                try:
                    if i == -1: browser.find_element(By.XPATH, '//li[@data-sort-method="NEW"]').click()
                    elif i == 0: browser.find_element(By.XPATH, '//li[@data-sort-method="BEST"]').click()
                    elif i > 0:
                        if i % 10 == 0: browser.find_element(By.XPATH, '//a[@class="edge_nav nav_next"]').click()
                        else: browser.find_element(By.XPATH, '//a[@class="num "][%d]'%(i%10)).click()
                    wait.until(EC.invisibility_of_element((By.CLASS_NAME, 'product_list_cover')))
                except: continue
                
                try:
                    productListDiv = browser.find_element(By.XPATH, '//div[@class="main_prodlist main_prodlist_list"]')
                    products = productListDiv.find_elements(By.XPATH, '//ul[@class="product_list"]/li')
                except: continue

                for product in products:
                    # 광고 상품 제거 로직
                    if not product.get_attribute('id'): continue
                    if 'prod_ad_item' in product.get_attribute('class').split(' '): continue
                    if product.get_attribute('id').strip().startswith('ad'): continue

                    try:
                        productId = product.get_attribute('id')[11:]
                        productName = product.find_element(By.XPATH, './div/div[2]/p/a').text.strip()
                        
                        # --- [핵심 수정: 원문 텍스트 추출] ---
                        # 복잡한 if-else 다 빼고 원문만 가져옵니다.
                        try:
                            spec_text = product.find_element(By.CLASS_NAME, 'spec_list').text
                        except:
                            try:
                                spec_text = product.find_element(By.XPATH, './/div[@class="spec_list"]').text
                            except:
                                spec_text = ""
                        
                        clean_spec_text = spec_text.replace(',', ' ').replace('\n', ' ')
                        spec_csvWriter.writerow([productId, productName, clean_spec_text])
                        # ----------------------------------

                        # 가격 추출 로직 (기존과 동일)
                        productPrices = product.find_elements(By.XPATH, './div/div[3]/ul/li')
                        productPriceStr = ''
                        isMall = 'prod_top5' in product.find_element(By.XPATH, './div/div[3]').get_attribute('class').split(' ')
                        
                        if isMall:
                            for productPrice in productPrices:
                                if 'top5_button' in productPrice.get_attribute('class').split(' '): continue
                                if productPriceStr: productPriceStr += DATA_PRODUCT_DIVIDER
                                try:
                                    mallName = productPrice.find_element(By.XPATH, './a/div[1]').text.strip()
                                    if not mallName: mallName = productPrice.find_element(By.XPATH, './a/div[1]/span[1]').text.strip()
                                    price = productPrice.find_element(By.XPATH, './a/div[2]/em').text.strip()
                                    productPriceStr += f'{mallName}{DATA_ROW_DIVIDER}{price}'
                                except: continue
                        else:
                            for productPrice in productPrices:
                                if productPriceStr: productPriceStr += DATA_PRODUCT_DIVIDER
                                try:
                                    productType = productPrice.find_element(By.XPATH, './div/p').text.strip()
                                    productType = productType.replace('\n', DATA_ROW_DIVIDER)
                                    productType = self.RemoveRankText(productType)
                                    price = productPrice.find_element(By.XPATH, './p[2]/a/strong').text.strip()
                                    if productType: productPriceStr += f'{productType}{DATA_ROW_DIVIDER}{price}'
                                    else: productPriceStr += f'{price}'
                                except: continue
                        
                        crawlingData_csvWriter.writerow([productId, productName, productPriceStr])
                    except Exception: continue

        except Exception as e:
            print('Error - ' + crawlingName + ' ->')
            print(traceback.format_exc())
            self.errorList.append(crawlingName)
        
        finally:
            try: browser.quit()
            except: pass
            crawlingFile.close()
            specTempFile.close()

        print('Crawling Finish : ' + crawlingName)

    def RemoveRankText(self, productText):
        if len(productText) < 2: return productText
        char1 = productText[0]
        char2 = productText[1]
        if char1.isdigit() and (1 <= int(char1) and int(char1) <= 9):
            if char2 == '위': return productText[2:].strip()
        return productText

    def DataSort(self):
        # 가격 데이터 병합 (기존 유지)
        print('Data Sort (Price History)\n')
        for crawlingValue in self.crawlingCategory:
            dataName = crawlingValue[STR_NAME]
            crawlingDataPath = f'{dataName}.csv'

            if not os.path.exists(crawlingDataPath): continue

            crawl_dataList = list()
            dataList = list()
            
            with open(crawlingDataPath, 'r', newline='', encoding='utf8') as file:
                csvReader = csv.reader(file)
                for row in csvReader: crawl_dataList.append(row)
            
            if len(crawl_dataList) == 0: continue
            
            dataPath = f'{DATA_PATH}/{dataName}.csv'
            if not os.path.exists(dataPath):
                file = open(dataPath, 'w', encoding='utf8')
                file.close()
                
            with open(dataPath, 'r', newline='', encoding='utf8') as file:
                csvReader = csv.reader(file)
                for row in csvReader: dataList.append(row)
            
            if len(dataList) == 0: dataList.append(['Id', 'Name'])
                
            dataList[0].append(crawl_dataList[0][0])
            dataSize = len(dataList[0])
            
            for product in crawl_dataList:
                if not str(product[0]).isdigit(): continue
                isDataExist = False
                for data in dataList:
                    if data[0] == product[0]:
                        if len(data) < dataSize: data.append(product[2])
                        isDataExist = True
                        break
                if not isDataExist:
                    newDataList = ([product[0], product[1]])
                    for i in range(2,len(dataList[0])-1): newDataList.append(0)
                    newDataList.append(product[2])
                    dataList.append(newDataList)
            
            for data in dataList:
                if len(data) < dataSize:
                    for i in range(len(data),dataSize): data.append(0)
            
            productData = dataList.pop(0)
            dataList.sort(key= lambda x: x[1])
            dataList.insert(0, productData)
                
            with open(dataPath, 'w', newline='', encoding='utf8') as file:
                csvWriter = csv.writer(file)
                for data in dataList: csvWriter.writerow(data)
                file.close()
                
            if os.path.isfile(crawlingDataPath): os.remove(crawlingDataPath)

    def DataSpecSave(self):
        # 스펙 원문 병합 (ID, Name, Spec_Details)
        print('Data Spec Save\n')
        for crawlingValue in self.crawlingCategory:
            dataName = crawlingValue[STR_NAME]
            tempSpecPath = f'{dataName}_Spec_Temp.csv'
            targetSpecPath = f'{SPEC_DATA_PATH}/{dataName}_Spec.csv'

            if not os.path.exists(tempSpecPath): continue
            
            saved_ids = set()
            if os.path.exists(targetSpecPath):
                with open(targetSpecPath, 'r', encoding='utf8') as f:
                    reader = csv.reader(f)
                    next(reader, None) 
                    for row in reader: 
                        if row: saved_ids.add(row[0])

            new_specs = []
            with open(tempSpecPath, 'r', encoding='utf8') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if not row: continue
                    p_id = row[0]
                    if p_id not in saved_ids:
                        new_specs.append(row)
                        saved_ids.add(p_id)

            with open(targetSpecPath, 'a', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                if os.path.getsize(targetSpecPath) == 0:
                    writer.writerow(['ID', 'Name', 'Spec_Details'])
                
                for row in new_specs: writer.writerow(row)

            if os.path.isfile(tempSpecPath): os.remove(tempSpecPath)
    
    def DataRefresh(self):
        # 매월 1일 데이터 백업 로직
        dTime = self.GetCurrentDate()
        if dTime.day == 1:
            print('Data Refresh\n')
            if not os.path.exists(DATA_PATH): os.mkdir(DATA_PATH)
            
            dTime -= timedelta(days=1)
            dateStr = dTime.strftime('%Y-%m')
            dataSavePath = f'{DATA_REFRESH_PATH}/{dateStr}'
            
            if not os.path.exists(DATA_REFRESH_PATH): os.mkdir(DATA_REFRESH_PATH)
            if not os.path.exists(dataSavePath): os.mkdir(dataSavePath)
            
            for file in os.listdir(DATA_PATH):
                fileName, fileExt = os.path.splitext(file)
                if fileExt == '.csv':
                    shutil.move(f'{DATA_PATH}/{file}', f'{dataSavePath}/{file}')
    
    def GetCurrentDate(self):
        tz = timezone(TIMEZONE)
        return (datetime.now(tz))

    def CreateIssue(self):
        # 깃허브 이슈 생성 로직
        if len(self.errorList) > 0:
            try:
                g = Github(os.environ.get(GITHUB_TOKEN_KEY, 'TOKEN_HERE'))
                repo = g.get_repo(GITHUB_REPOSITORY_NAME)
                title = f'Crawling Error - ' + self.GetCurrentDate().strftime('%Y-%m-%d')
                body = ''
                for err in self.errorList: body += f'- {err}\n'
                labels = [repo.get_label('bug')]
                repo.create_issue(title=title, body=body, labels=labels)
            except: pass

if __name__ == '__main__':
    crawler = DanawaCrawler()
    crawler.DataRefresh()
    crawler.StartCrawling()
    crawler.DataSort()
    crawler.DataSpecSave()
    crawler.CreateIssue()