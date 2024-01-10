from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import requests
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager

chrmoe_release = "https://chromedriver.storage.googleapis.com/LATEST_RELEASE"
chrmoe_version = requests.get(chrmoe_release).text
global_url = 'https://www.gsp.or.kr'
gsp_url_list = [#'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=Z1',
                'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=A1', 
                'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=A2',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=B1',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=B2',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=C1',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=C2',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=C3',
                #'https://www.gsp.or.kr/startUp/UVUL0001.do?searchFlag=D1'
                ]


data_dic = ['sector', 'company_name', 'name', 'keyword', 'homepage', 'addr']
company_data_list = []

def x_path_addr(num,index):
    return '//*[@id="folio"]/div/div/div[3]/div/div[2]/div[{}]/div/div/div[{}]'.format(num, index)

def write_excel(data_list):
    dfs = [pd.DataFrame([data]) for data in data_list]

    df = pd.concat(dfs,ignore_index=True)

    df.to_excel('output.xlsx', index=False)

def search_main():

    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(ChromeDriverManager(version=chrmoe_version).install(),  options=options)
    sub_driver =  webdriver.Chrome(ChromeDriverManager(version=chrmoe_version).install(),  options=options)

    for gsp_url in gsp_url_list:
        #try:
            driver.get(gsp_url)
            WebDriverWait(driver,60).until(EC.presence_of_element_located((By.XPATH,'//*[@id="folio"]/div/div/div[3]/div/div[1]/div[1]/h3')))
            num = int(driver.find_element_by_id('startupListCount').text.replace(',',''))
            
            #드롭다운메뉴 선택
            #drop_dw_btn = driver.find_element_by_xpath('//*[@id="list_num_select"]')
            #select = Select(drop_dw_btn)
            #select.select_by_value('o1')
            #WebDriverWait(driver,60).until(EC.presence_of_element_located((By.XPATH,'//*[@id="folio"]/div/div/div[3]/div/div[1]/div[1]/h3')))
            #print(num-1000)
            for i in range(1,num-1000):
                try:
                    x_path_str = x_path_addr(i,3)
                    overlay_addr = driver.find_element_by_xpath(x_path_str)
                    sub_url = overlay_addr.find_element_by_tag_name("a").get_attribute('onclick').split("'")[1]

                    sub_driver.get(global_url+sub_url)
                    WebDriverWait(sub_driver,60).until(EC.presence_of_element_located((By.XPATH,'//*[@id="contents"]/div[3]/div/div[2]')))
                    
                    table = sub_driver.find_element_by_class_name('wordntable')
                    tbody = table.find_element_by_tag_name("tbody")
                    rows = tbody.find_elements_by_tag_name("tr")
                    
                    company_data_dic = {'sector' : '', 'company_name' : '', 'name' : '', 'keyword' : '', 'homepage' : '', 'addr' : ''}

                    for index, value in enumerate(rows):
                        company_data_dic[data_dic[index]]=value.find_elements_by_tag_name("td")[0].text


                    company_data_list.append(company_data_dic)
                    
                except:
                    print(gsp_url, i," sub exception!!!")

        #except:
        #    print(gsp_url," exception!!!")
    write_excel(company_data_list)
    sub_driver.close()
    driver.close()

def job():

    search_main()

if __name__ == "__main__":
    job()