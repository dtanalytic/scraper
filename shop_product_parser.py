
from abc import ABCMeta, abstractmethod
from bs4 import BeautifulSoup
import random
import time
import re
import os
import requests
import pickle
from datetime import datetime,timedelta
import locale

import sys
sys.path.append('../')
from general_modules import net_scrape


class StopException(Exception):
    pass

class NoMoreNewRecordsException(Exception):
    pass

class ShopProductsParser(metaclass=ABCMeta):

    # cur_url, url_base, url_add, params, tag_container_el,tag_el, products_list, num_records_pass_in_page pages_load_stop records_ignore

    def initClassVariables(self, cur_url, tag_container_el, tag_el, page_param, url_alt, pages_load_stop):
        self.cur_url = cur_url
        self.page_param = page_param
        self.tag_container_el= tag_container_el if isinstance(tag_container_el,dict) else {'name':tag_container_el.split(',')[0],'class':tag_container_el.split(',')[1]}
        self.tag_el=tag_el if isinstance(tag_el,dict) else {'name':tag_el.split(',')[0],'class':tag_el.split(',')[1]}
        self.url_alt=url_alt
        self.products_list = []
        self.products_list_html_json = []
        self.num_records_pass_in_page = 0
        self.pause_flag=False
        self.url_base, self.url_add, self.params, = self.parseUrl(cur_url)
        self.pages_load_stop=pages_load_stop
        self.records_ignore=0

    def __init__(self,cur_url, tag_container_el,tag_el,page_param, pages_load_stop=10000, url_alt=''):
        self.initClassVariables(cur_url, tag_container_el,tag_el,page_param,url_alt, pages_load_stop)


    def getListFrom2Tags(self,url,tag_container,tag):
        try:
            tag_list =''
            #html = self.afterConnectionDelay(url).text
            html = net_scrape.get_url_delay(self.delay, url).text
            bsObj = BeautifulSoup(html, 'lxml')
            tag_list = bsObj.find(tag_container['name'],class_=tag_container['class']).find_all(tag['name'],class_=tag['class'])

        finally:
            if not tag_list:
                print('поиск двух контейнеров прошел неудачно')
                #return ''
                raise StopException
            else: return tag_list

    def parseUrl(self,cur_url):
        re_parts = re.search('(http\w{0,1}://www.*\..{1,4})(/.*\?)(.*)',cur_url)
        url_base,url_add, url_params=re_parts.groups()
        params=''
        if not url_params=='':
            url_params=url_params.split('&')
            params = {param.split('=')[0]:param.split('=')[1] for param in url_params}

        return url_base, url_add, params

    def makeUrlFromParts(self,url_base,url_add,params):
        str_param = '&'.join([key + '=' + val for (key, val) in params.items()])
        return url_base+url_add+str_param

    @abstractmethod
    def getNextURL(self,url,tag_container,tag):
        pass

    @abstractmethod
    def filterProductAcToSite(self,product_params):
        pass

    def startListProductsParser(self, messages_queue, product_last_datetime):
       i=1
       #max_sec=7
       time1 = time.time()
       try:
            while self.cur_url !='':# and i<=max_sec:

              if not self.pause_flag and i<= int(self.pages_load_stop):
                # если ответ в формате html
                if self.url_alt:
                    tag_list = self.getListRefJson(self.cur_url,self.tag_el)
                else:
                    tag_list = self.getListFrom2Tags(self.cur_url,self.tag_container_el,self.tag_el)

                #a = tag_list
                #set_trace()
                self.products_list_html_json=tag_list

                self.getProductItems(messages_queue, product_last_datetime)


                time2 = time.time()
               # print('закончили {} страницу , ее обработка заняла {} секунд'.format(i, time2-time1))
                messages_queue.put('закончили {} страницу , ее обработка заняла {} секунд'.format(self.params[self.page_param], time2-time1))
                self.cur_url = self.getNextURL(self.cur_url, self.tag_container_pages, self.tag_page)

                self.num_records_pass_in_page=0
                i = i + 1
              else:
                  if i>int(self.pages_load_stop):
                      self.pause_flag = True

                      messages_queue.put('change ip')

                  raise StopException

       except NoMoreNewRecordsException:
           messages_queue.put('no more new records')
       finally:
            pass

    def getProductItems(self, messages_queue, product_last_datetime):
       #i=0
       #max_sec=60
       start = self.num_records_pass_in_page

       if not self.url_alt:
            href_parts=[product_html_json.find('a', class_=self.PRODUCT_ITEM_HREF_CLASS).get('href') for product_html_json in self.products_list_html_json]
       else:
            href_parts = self.products_list_html_json
       #href_parts.sort()
       try:
           #for j in range(0, 5):
            for j in range(start, len(href_parts)):
              if not self.pause_flag:
               #if i < max_sec:
                time1 = time.time()
                if self.url_alt:
                    re_parts = re.search('(.*-)(.*)',href_parts[j])
                    __, url_add = re_parts.groups()
                    url = self.url_alt +url_add
                else:
                    url = self.url_base + href_parts[j]


                product_params = self.product_params_init.copy()
                self.getWholeProductParams(product_params, url)
                # нужно старое и новое значение, чтобы не добавлять эту запись если надо игнорить
                records_ignore_was=self.records_ignore

                if not product_last_datetime=='':
                    locale.setlocale(locale.LC_ALL, '')
                    #a = self.makeDateStb(product_params['date_time'])

                    if self.makeDateStb(product_params['date_time'])<=product_last_datetime:
                        #set_trace()
                        self.records_ignore= self.records_ignore+1
                        messages_queue.put('ignore {}'.format(self.records_ignore))

                    # locale.setlocale(locale.LC_ALL, 'en')
                    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
                    if self.records_ignore>9: raise NoMoreNewRecordsException


                # если не надо игнорировать то добавляем
                if records_ignore_was == self.records_ignore and self.filterProductAcToSite(product_params):
                    self.products_list.append(product_params)

                self.num_records_pass_in_page += 1
                time2 = time.time()
                messages_queue.put('{} запись, ее обработка заняла {} секунд'.format(j,time2-time1))
                #print('{} запись, ее обработка заняла {} секунд'.format(j,time2-time1))
                #i = i + 1
              else:
                raise StopException
       finally:
            pass


    def makeDateStb(self, x):

            month_dict = {'января': 'январь', 'февраля': 'февраль', 'марта': 'март', 'апреля': 'апрель', 'мая': 'май',
                          'июня': 'июнь', \
                          'июля': 'июль', 'августа': 'август', 'сентября': 'сентябрь', 'октября': 'октябрь', \
                          'ноября': 'ноябрь', 'декабря': 'декабрь'}

            # for k, v in month_dict.items():
            #     if x.lower().find(k) != -1:
            #         x = x.lower().replace(k, v)
            #         break
            for k, v in month_dict.items():
                if x.lower().find(v) != -1:
                    x = x.lower().replace(v,k)
                    break

            fmt = '%d %B %H:%M %Y'
            datetime_t = time.strptime(str(x), fmt)
            datetime_socr = datetime(datetime_t.tm_year, datetime_t.tm_mon, datetime_t.tm_mday, datetime_t.tm_hour,datetime_t.tm_min)
            return datetime_socr


    def getClassParams(self):
        class_params_init = {}
        class_params_init['cur_url'] = self.cur_url
        class_params_init['num_records_pass_in_page'] = self.num_records_pass_in_page
        class_params_init['cur_page'] = self.params[self.page_param]
        class_params_init['last_date'] = self.products_list[-1]['date_time'] if not len(self.products_list)==0 else ''
        class_params_init['cost'] = self.products_list[-1]['cost'] if not len(self.products_list) == 0 else ''

        return class_params_init

    def setClassParams(self, tuple_params):

        self.products_list = []
        self.cur_url = tuple_params['cur_url']
        self.params[self.page_param] = tuple_params['cur_page']
        self.pause_flag = False
        self.num_records_pass_in_page = tuple_params['num_records_pass_in_page']




if __name__=='__main__':

    import preferences_window
    page_parser = preferences_window.Window.getSiteClass()

    cur_url = page_parser.getNextURL(page_parser.cur_url, page_parser.tag_container_pages, page_parser.tag_page)

    # list_tags = page_parser.getListFrom2Tags(page_parser.cur_url, page_parser.tag_container_pages, page_parser.tag_page)
    # if list_tags:
    #     pass
    #     pages_count = list_tags[-2].text
    #     url_base, url_add, params = page_parser.parseUrl(page_parser.cur_url)
    #
    #
    #     current_page_num = page_parser.params[page_parser.page_param]
    #     if int(current_page_num) + 1 <= int(pages_count):
    #         page_parser.params[page_parser.page_param] = str(int(current_page_num) + 1)
    #        a= page_parser.makeUrlFromParts(page_parser.url_base, page_parser.url_add, page_parser.params)


    # html = net_scrape.get_url_delay(1, 'https://www.avito.ru/vladikavkaz/kvartiry/prodam-ASgBAgICAUSSA8YQ?p=1').text
    # bsObj = BeautifulSoup(html, 'lxml')

    # tag_container_el = 'div', 'js-catalog_serp'
    # tag_el = 'div', 'item_table'
    # tag_container_pages = 'div', 'pagination-root-2oCjZ'

    # tag_page = 'span', 'pagination-item-1WyVp'


    # tag_list = bsObj.find(tag_container_pages[0], class_=tag_container_pages[1]).find_all(tag_page[0], class_=tag_page[1])
    
    
    
    
    