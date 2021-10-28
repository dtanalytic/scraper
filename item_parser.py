import sys
sys.path.append('../')
from general_modules import net_scrape
from general import StopException,NoMoreNewRecordsException
from general import PAGES_LOAD_STOP_NUM,REC_IGN_BEF_STOP_MAX
from bs4 import BeautifulSoup
from bs4.element import Tag
from abc import ABCMeta, abstractmethod
import random
import time
import re
import numpy as np
import requests
import pickle
from datetime import datetime,timedelta




    
class ItemsParser(metaclass=ABCMeta):    
        
    def __init__(self,cur_url, tag_container_el,tag_el, rec_ign_bef_stop_max = 
                 REC_IGN_BEF_STOP_MAX, pages_load_stop_num=PAGES_LOAD_STOP_NUM):
        self.cur_url = cur_url
        # строчные описания для тега контейнера ссылок и тегов,
        # в которых непосредственно хранятся ссылки 
        self.tag_container_el = tag_container_el
        self.tag_el=tag_el
        # переменная хранит собранные записи
        self.items_list = []
        # подсчет количества собранных записей
        self.records_pass_in_page_num = 0
        # используется, если посылаем останов извне
        self.pause_flag=False
        # после стольки страниц скачивание
        # останавливается автоматически
        # сделано, чтобы много с одного ip не скачивать 
        # (а то заблокировать могут) и переключиться на другой
        self.pages_load_stop_num = pages_load_stop_num
        # иногда старые объявления поднимаются
        # наверх, поэтому для критерия останова
        # надо пройти несколько старых записей,
        # чтобы убедиться в отсутствии новых 
        self.rec_ign_bef_stop_num = 0
        self.rec_ign_bef_stop_max = rec_ign_bef_stop_max

    @classmethod
    def get_items_list_from2tags(cls,url_bs4,tag_container,tag,delay):
        
        tag_container = tag_container if isinstance(tag_container,dict)\
            else {'name':tag_container.split(',')[0],\
                  'field':tag_container.split(',')[1],\
                  'value':tag_container.split(',')[2]}
        tag=tag if isinstance(tag,dict) \
            else {'name':tag.split(',')[0],\
                  'field':tag.split(',')[1],\
                  'value':tag.split(',')[2]}
        
        tag_list =''
        
        try:
            # если страница уже скачивалась, то опять не качаем и вместо url
            # передаем ее в виде классов либо BeautifulSoup либо Tag
            if not isinstance(url_bs4,BeautifulSoup) and not isinstance(url_bs4,Tag):
                html = net_scrape.get_url_delay(delay, url_bs4).text
                bsObj = BeautifulSoup(html, 'lxml')
            else: bsObj = url_bs4
            tag_list = bsObj\
                .find(tag_container['name'],{tag_container['field']:tag_container['value']})\
                .find_all(tag['name'],{tag['field']:tag['value']})

        finally:
            if not tag_list:
                print('поиск двух контейнеров прошел неудачно')                
                raise StopException
                
            else: return tag_list, bsObj
    
    def parse_url(self,cur_url):
        re_parts = re.search('(http\w{0,1}://w{0,3}.*\..{1,4})(/.*\?{1})(.*)',cur_url)
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
    def get_next_url(self):
        pass
    
    @abstractmethod
    def get_one_item_params(self):
        pass


    
    def start_items_parser(self, messages_queue = None, item_last_datetime = ''):
       i=0       
       time1 = time.time()
       try:
           # self.cur_url хранит адрес текущей страницы 
            while self.cur_url !='':
              if not self.pause_flag and i <= int(self.pages_load_stop_num):
                # получает ссылки на товары/события с текущей страницы,
                # реализуется в специфическом модуле, так как
                # зависит от конкретного сайта
                items_hrefs = self.get_item_hrefs(self.cur_url,self.tag_container_el,
                                                  self.tag_el, self.delay)
                # собирает записи по полученным ссылкам, реализована ранее
                self.get_items_params(list(set(items_hrefs)), messages_queue, 
                                      item_last_datetime)
                
                if messages_queue:
                    time2 = time.time()
                    messages_queue.put('''закончили итерацию по {} странице, 
                                       ее обработка заняла {} секунд'''\
                                       .format(self.params[self.page_param], 
                                               time2-time1))
                # получение url новой страницы,
                # реализуется в специфическом модуле
                self.cur_url = self.get_next_url()
                self.records_pass_in_page_num = 0                
                i = i + 1
              else:
                  # после стольки страниц принудительный останов и смена ip
                  if i>int(self.pages_load_stop_num):
                      self.pause_flag = True
                      if messages_queue:
                          messages_queue.put('change ip')
                  raise StopException
       # это исключение генерирует get_items_params,
       # если новых записей нет
       except NoMoreNewRecordsException:
           if messages_queue:
               messages_queue.put('no more new records')
       finally:
            pass
        
    def get_items_params(self, items_hrefs, messages_queue = None, item_last_datetime=''):
       start = self.records_pass_in_page_num 
       try:           
            for j in range(start, len(items_hrefs)):
              if not self.pause_flag:               
                time1 = time.time()
                url = items_hrefs[j]
                # словарь со значениями одной записи
                item_params = self.get_one_item_params(url)                
                # для проверки игнорируется ли текущая запись
                records_ignore_was = self.rec_ign_bef_stop_num
                if not item_last_datetime == '':                    
                    if item_params['Date']<=item_last_datetime:
                        self.rec_ign_bef_stop_num = self.rec_ign_bef_stop_num+1
                        if messages_queue:
                            messages_queue.put('ignore {}'.format(self.rec_ign_bef_stop_num))
                    if self.rec_ign_bef_stop_num>self.rec_ign_bef_stop_max: 
                        raise NoMoreNewRecordsException
                # если не надо игнорировать то добавляем
                if records_ignore_was == self.rec_ign_bef_stop_num:
                     self.items_list.append(item_params)

                self.records_pass_in_page_num += 1
                if messages_queue:
                    time2 = time.time()
                    messages_queue.put('{} запись, ее обработка заняла {} секунд'
                                       .format(j,time2-time1))
                
              else:
                raise StopException
       finally:
           pass
            
    


class YoulaProductParser(ItemsParser):
    
    def __init__(self,cur_url, tag_container_el,tag_el, rec_ign_bef_stop_max=REC_IGN_BEF_STOP_MAX, pages_load_stop_num=PAGES_LOAD_STOP_NUM):
        super().__init__(cur_url, tag_container_el,tag_el, rec_ign_bef_stop_max, pages_load_stop_num)
        
    def get_next_url(self,url,tag_container,tag):
        pass
    
    def get_one_item_params(self, url):
        pass
    
    def get_item_hrefs(self, tag_list):          
        return [self.url_base + item.find('a').get('href') for item in tag_list]   
    

    
if __name__=='__main__':
    
    
    from ufc.ufc_fights_parser import UFCFightsParser
    # адрес www и ?
    cur_url = 'http://www.ufcstats.com/event-details/a1153013cb5f628f?'
    events_url = 'http://www.ufcstats.com/statistics/events/completed?page=1'  
    tag_container_events = 'tbody,,'
    tag_event = 'a,class,b-link b-link_style_black'
    tag_container_el = 'tbody,class,b-fight-details__table-body'
    tag_el = 'a,class,b-flag'
    page_param = 'page'
    delay = 1
    ufc_fights = UFCFightsParser(cur_url,events_url,delay,page_param,tag_container_events, tag_event, tag_container_el,tag_el)
    
    # saved_d = {

    #     'cur_url':'http://www.ufcstats.com/event-details/5df17b3620145578?',
    #     # номер страницы с турниром на сайте
    #     # задается в виде строки для внутрен. манипуляций
    #     'cur_page':'2',
    #     # номер записи с которой начнется скачивание
    #     'records_pass_in_page_num':2,
    #     # полный url страницы с турнирами
    #     'events_url':'http://www.ufcstats.com/statistics/events/completed?page=2',
    #     # номер турнира на странице ссылок на турниры
    #     'event_ind':0,
    #     'events_hrefs':['http://www.ufcstats.com/event-details/5df17b3620145578'],
    #     'event_date':'February 15, 2020',
    #     'event_place':'vlad',
    #     'event_attendence':1212
    #             }
    
    # ufc_fights.load_class_params(saved_d)


    # так качаем конкретный event
    # events_url и event_url
    items_hrefs = ufc_fights.get_item_hrefs(ufc_fights.cur_url,\
                ufc_fights.tag_container_el,ufc_fights.tag_el, ufc_fights.delay)

    ufc_fights.get_items_params(list(set(items_hrefs)))
    
    items_list = ufc_fights.items_list

    import pandas as pd
    frame = pd.DataFrame(items_list)
    frame.to_csv('one_event.csv', index=False)    
    
    frame_new = pd.read_csv('one_event.csv')
    frame_old = pd.read_csv('items.csv')
    
    # frame_old1 = frame_old.iloc[:5716]
    
    # frame_old2 = frame_old.iloc[5716:]
    # frame_old = pd.concat([frame_old1,frame_new,frame_old2], ignore_index=True)
    
    # frame_old.to_csv('ufc_fights.csv', index=False) 
    
    frame = pd.concat([frame_new,frame_old], ignore_index=True)
    frame.to_csv('items.csv', index=False)
    
    
    
    
    
    # delay = 1
    # url = 'http://www.ufcstats.com/event-details/805ad1801eb26abb'
    # html = net_scrape.get_url_delay(delay=delay, url = url).text
    # bsObj = BeautifulSoup(html, 'lxml')

    # l = bsObj.find_all('a',{'class':'b-flag'})


    
    # tag_container_el = 'tbody,class,b-fight-details__table-body'
    # tag_el = 'a,class,b-flag b-flag_style_green'
    
    # tag_container_el='ul,class,product_list _board_items'
    # tag_el='li,class,product_item'
    # page_param='page'
    # cur_url='https://youla.ru/vladikavkaz/nedvijimost/prodaja-kvartiri?page=1'
        
    # youla_products = YoulaProductParser(cur_url, tag_container_el,tag_el)
    # youla_products.delay = 1
    
    # youla_products.start_items_parser()
    
    
