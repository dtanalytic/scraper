

from datetime import datetime,timedelta
from bs4 import BeautifulSoup
import numpy as np
import sys
sys.path.append('../')
from general_modules import net_scrape
from general import StopException,NoMoreNewRecordsException
from general import PAGES_LOAD_STOP_NUM,REC_IGN_BEF_STOP_MAX
from general import set_eng_locale
from item_parser import ItemsParser
from nltk import word_tokenize

# delay = 1
# UFC_FIGHT_TAGS = [('a','class','b-link b-link_style_black')
#                   ]



class UFCFightsParser(ItemsParser):
    
    EVENTS_DATES_TAG_STR = 'span,class,b-statistics__date'

    
    def __init__(self,cur_url, events_url, delay, page_param, tag_container_pages,tag_page, tag_container_el, tag_el, rec_ign_bef_stop_max=REC_IGN_BEF_STOP_MAX, pages_load_stop_num=PAGES_LOAD_STOP_NUM):
        
        
        
        self.tag_container_pages = tag_container_pages if isinstance(tag_container_pages,dict)\
            else {'name':tag_container_pages.split(',')[0],\
                  'field':tag_container_pages.split(',')[1],\
                  'value':tag_container_pages.split(',')[2]}
        self.tag_page = tag_page if isinstance(tag_page,dict)\
            else {'name':tag_page.split(',')[0],\
                  'field':tag_page.split(',')[1],\
                  'value':tag_page.split(',')[2]}
        self.events_url = events_url
        self.page_param = page_param
        # url разбиваем на части
        self.url_base, self.url_add, self.params = self.parse_url(events_url)
        self.delay=delay
        
        super().__init__(cur_url, tag_container_el,tag_el, rec_ign_bef_stop_max, pages_load_stop_num)
        
        events_hrefs_tags = self.get_items_list_from2tags(self.events_url,self.tag_container_pages,self.tag_page)
        self.events_hrefs = [item.attrs['href'] for item in events_hrefs_tags]
        self.event_ind = self.events_hrefs.index(self.cur_url[:-1])
        events_dates_tag_dict = {'name':self.EVENTS_DATES_TAG_STR.split(',')[0],'field':self.EVENTS_DATES_TAG_STR.split(',')[1],'value':self.EVENTS_DATES_TAG_STR.split(',')[2]}
        tag_dates = self.get_items_list_from2tags(self.events_url,self.tag_container_pages,events_dates_tag_dict)
        self.events_dates = [item.get_text().strip()  for item in tag_dates]
        
        
    def get_next_URL(self):
        
         if self.event_ind  == self.events_num-1:
                self.params[self.page_param]=str(int(self.params[self.page_param])+1)
                self.events_url = self.make_url_from_parts(self.url_base, self.url_add, self.params)
                
                tag_list = self.get_items_list_from2tags(self.events_url,self.tag_container_pages,self.tag_page)
                if tag_list:
                    self.events_hrefs = [item.attrs['href'] for item in tag_list]
                    self.event_ind = 0
                    return self.events_hrefs[self.event_ind]+"?"

         else:
            self.event_ind = self.event_ind+1
            return self.events_hrefs[self.event_ind]+'?'
        
        
        
    
    def get_item_hrefs(self, tag_list):        
        return [item.attrs['href'] for item in tag_list]
    
    def get_one_item_params(self, url):
        item_params = {}
        event_date_str = self.events_dates[self.event_ind]
        with set_eng_locale():
            event_date = datetime.strptime(event_date_str,'%B %d, %Y')
            
            
        # dates = [datetime.now() - timedelta(days=1),
        #          datetime.now() - timedelta(days=2),
        #          datetime.now() - timedelta(days=3),
        #          datetime.now() - timedelta(days=4),
        #          datetime.now() - timedelta(days=5)]
        
        item_params['date'] = event_date
        
        return item_params

if __name__ == '__main__':
    
    # cur_url = 'http://www.ufcstats.com/event-details/542db012217ecb83?'
    # events_url='http://www.ufcstats.com/statistics/events/completed?page=1'
    # tag_container_pages='tbody,,'
    # tag_page='a,class,b-link b-link_style_black'
    # tag_container_el = 'tbody,class,b-fight-details__table-body'
    # tag_el = 'a,class,b-flag b-flag_style_green'
    # page_param = 'page'
    # delay = 1
    # ufc_fights = UFCFightsParser(cur_url,events_url,delay,page_param,tag_container_pages, tag_page, tag_container_el,tag_el)
    
    
    
    
    delay = 1
    url = 'http://www.ufcstats.com/fight-details/524b49a676498c6d'
    html = net_scrape.get_url_delay(delay=delay, url = url).text
    bsObj = BeautifulSoup(html, 'lxml')
     
    ev_name = bsObj.find('a',{'class':'b-link'}).get_text().strip()
    
    f_names_t = bsObj.findAll('h3',{'class':'b-fight-details__person-name'})
    f_names = [tag.get_text().strip() for tag in f_names_t]
    
    f_res_t = bsObj.findAll('i',{'class':'b-fight-details__person-status'})
    f_res = [tag.get_text().strip() for tag in f_res_t]
    
      
    f_det_res_tags = bsObj.findAll('i',{'class':'b-fight-details__label'})
   
    det_res = {}
    for i,f_det_res_tag in enumerate(f_det_res_tags):
        key = f_det_res_tag.get_text().strip()
        det_res[key] = f_det_res_tag.parent.get_text().strip().replace(key,'').strip()
    
        
    total_res = bsObj.findAll('th',{'class':'b-fight-details__table-col'})
    [item.get_text().strip() for item in total_res]    
    
    # tags  = bsObj.findAll('a',{'style':'color:#B10101'})
    
    
    # ufc_fights.get_next_URL(cur_url, tag_container_pages, tag_page)
    
    
    
    
    
    
    
    
    
    
    
    
    