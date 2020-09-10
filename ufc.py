import sys
sys.path.append('../')
from general_modules import net_scrape
from bs4 import BeautifulSoup
from abc import ABCMeta, abstractmethod
import random
import time
import re
import os
import requests
import pickle
from datetime import datetime,timedelta
import locale



delay = 1
UFC_FIGHT_TAGS = [('a','class','b-link b-link_style_black')
                  ]


class StopException(Exception):
    pass

class NoMoreNewRecordsException(Exception):
    pass


    
class ItemsParser(metaclass=ABCMeta):
    
        
    def __init__(self,cur_url, tag_container_el,tag_el, pages_load_stop=10000):
        self.cur_url = cur_url
        self.tag_container_el= tag_container_el if isinstance(tag_container_el,dict) else {'name':tag_container_el.split(',')[0],tag_container_el.split(',')[1]:tag_container_el.split(',')[2]}
        self.tag_el=tag_el if isinstance(tag_el,dict) else {'name':tag_el.split(',')[0],tag_el.split(',')[1]:tag_el.split(',')[2]}
        self.items_list = []        
        self.num_records_pass_in_page = 0
        # если посылаем останов извне
        self.pause_flag=False
        # url разбиваем на части
        self.url_base, self.url_add, self.params, = self.parse_url(cur_url)
        self.pages_load_stop=pages_load_stop
        # иногда старые объявления поднимаются наверх, поэтому для критерия останова
        # надо пройти несколько старых записей, чтобы убедиться, что новых нет
        self.records_ignore=0
        

    def get_items_list_from2tags(self,url,tag_container,tag):
        try:
            tag_list =''
            html = net_scrape.get_url_delay(self.delay, url).text
            bsObj = BeautifulSoup(html, 'lxml')
            tag_list = bsObj.find(tag_container['name'],class_=tag_container['class']).find_all(tag['name'],class_=tag['class'])

        finally:
            if not tag_list:
                print('поиск двух контейнеров прошел неудачно')
                #return ''
                raise StopException
            else: return tag_list
    
    def parse_url(self,cur_url):
        re_parts = re.search('(http\w{0,1}://www.*\..{1,4})(/.*\?)(.*)',cur_url)
        url_base,url_add, url_params=re_parts.groups()
        params=''
        if not url_params=='':
            url_params=url_params.split('&')
            params = {param.split('=')[0]:param.split('=')[1] for param in url_params}

        return url_base, url_add, params
    

    def make_url_from_parts(self,url_base,url_add,params):
        str_param = '&'.join([key + '=' + val for (key, val) in params.items()])
        return url_base+url_add+str_param

    @abstractmethod
    def get_next_URL(self,url,tag_container,tag):
        pass
    
    
    def start_items_parser(self, product_last_datetime = '', messages_queue = None):
       i=0
       max_items=1
       try:
            while self.cur_url !='' and i<=max_items: # and i<=max_items:

              if not self.pause_flag and i<= int(self.pages_load_stop):
                # если ответ в формате html
                # if self.url_alt:
                #     tag_list = self.getListRefJson(self.cur_url,self.tag_el)
                # else:
                self.items_list = self.get_items_list_from2tags(self.cur_url,self.tag_container_el,self.tag_el)
                self.items_hrefs = self.get_item_hrefs(self.items_list)
                # self.get_items_params(messages_queue, product_last_datetime)


                # messages_queue.put('закончили {} страницу , ее обработка заняла {} секунд'.format(self.params[self.page_param], time2-time1))
                
                # self.cur_url = self.get_next_URL(self.cur_url, self.tag_container_pages, self.tag_page)

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
        
    def get_items_params(self, messages_queue, product_last_datetime):
       #i=0
       #max_sec=60
       start = self.num_records_pass_in_page

       # if not self.url_alt:
       #      href_parts=[product_html_json.find('a', class_=self.PRODUCT_ITEM_HREF_CLASS).get('href') for product_html_json in self.products_list_html_json]
       # else:
       #      href_parts = self.products_list_html_json
       
       #href_parts.sort()
       
       
       try:
           #for j in range(0, 5):
            for j in range(start, len(self.items_hrefs)):
              if not self.pause_flag:
               #if i < max_sec:
                time1 = time.time()
                url = self.items_hrefs[j]
                # if self.url_alt:
                #     re_parts = re.search('(.*-)(.*)',href_parts[j])
                #     __, url_add = re_parts.groups()
                #     url = self.url_alt +url_add
                # else:
                #     url = self.url_base + href_parts[j]


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
            
    
class UFCFightsParser(ItemsParser):
    
    def __init__(self,cur_url, tag_container_el,tag_el, pages_load_stop=10000):
        super().__init__(cur_url, tag_container_el,tag_el, pages_load_stop)
        
    def get_next_URL(self,url,tag_container,tag):
        pass
    
    def get_item_hrefs(self, tag_list):        
        return [item.attrs['href'] for item in tag_list]

class ULAProductParser(ItemsParser):
    
    def __init__(self,cur_url, tag_container_el,tag_el, pages_load_stop=10000):
        super().__init__(cur_url, tag_container_el,tag_el, pages_load_stop)
        
    def get_next_URL(self,url,tag_container,tag):
        pass
    
    def get_item_hrefs(self, tag_list):        
        return [item.attrs['href'] for item in tag_list]    
    
    
if __name__=='__main__':
    
    cur_url = 'http://www.ufcstats.com/event-details/542db012217ecb83?'
    tag_container_el = 'tbody,class,b-fight-details__table-body'
    tag_el = 'a,class,b-flag b-flag_style_green'
    
    ufc_fights = UFCFightsParser(cur_url, tag_container_el,tag_el)
    ufc_fights.delay = 1
    
    
    # hrefs = [item.attrs['href'] for item in tag_list]
    
    
    
    
    
    ufc_fights.start_items_parser()
    # delay = 1
    # url = 'http://ufcstats.com/statistics/events/completed'
    # html = net_scrape.get_url_delay(delay=delay, url = url).text
    # bsObj = BeautifulSoup(html, 'lxml')
    # tag_container_list = []    
    
    # hrefs = bsObj.find_all('a',{'class':'b-link b-link_style_black'})
    # hrefs = [item.attrs['href'] for item in hrefs]