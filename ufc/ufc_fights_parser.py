
from collections import OrderedDict
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
    

    def __init__(self,cur_url, events_url, delay, page_param, tag_container_events,tag_event, tag_container_el, tag_el, rec_ign_bef_stop_max=REC_IGN_BEF_STOP_MAX, pages_load_stop_num=PAGES_LOAD_STOP_NUM):
        
        # страница со ссылками на события/турниры
        self.events_url = events_url
        # строчные описания контейнера и внутренних тегов с ссылками 
        # на все события на странице events_url
        self.tag_container_events = tag_container_events 
        self.tag_event = tag_event
        # параметр, отвечающий в events_url 
        # за номер страницы, для UFC - 'page'
        self.page_param = page_param
        # parse_url - метод ItemsParser для разбиения url 
        # на части: адрес сайта, относительный путь
        # по папкам, параметры и их значения
        self.url_base, self.url_add, self.params = self.parse_url(events_url)
        # задержка при скачивании каждой страницы с сайта
        self.delay=delay
        # конструктор базового класса ItemsParser
        super().__init__(cur_url, tag_container_el,tag_el, rec_ign_bef_stop_max, pages_load_stop_num)
        events_hrefs_tags, _ = self.get_items_list_from2tags(self.events_url,self.tag_container_events,self.tag_event, self.delay)
        # ссылки на все события на странице events_url
        self.events_hrefs = [item.attrs['href'] for item in events_hrefs_tags]
        self.events_num = len(self.events_hrefs)
        # self.event_ind - индекс текущего события,
        # которое ищем путем поиска
        # его url в списке. Последний символ отсекаем, так как 
        # в настройках cur_url адрес задается с ? на конце
        # для простоты отделения частей адреса
        # от параметров в методе self.parse_url
        self.event_ind = self.events_hrefs.index(self.cur_url[:-1])
        # дата, место события, число зрителей
        self.event_date, self.event_place, self.event_attendence = self.get_event_inf(cur_url,'i,class,b-list__box-item-title') 
        
        
        
    # возвращает ссылку на новый турнир
    # для функции обхода start_items_parser
    def get_next_url(self):
         # если так, то надо переходить на новую страницу UFC
         if self.event_ind  == self.events_num-1:
                # self.params - словарь, хранящий параметры
                # и их значения в url
                self.params[self.page_param]=str(int(self.params[self.page_param])+1)
                # соединяем части для образования
                # ссылки на новую страницу UFC
                self.events_url = self.make_url_from_parts(self.url_base, self.url_add, self.params)
                # обновляем поля, описанные в конструкторе
                events_hrefs_tags, _ = self.get_items_list_from2tags(self.events_url,self.tag_container_events,self.tag_event, self.delay)
                if events_hrefs_tags:
                    self.events_hrefs = [item.attrs['href'] for item in events_hrefs_tags]
                    self.event_ind = 0
                    self.events_num = len(self.events_hrefs)

        # если турниры в рамках данной страницы
        # еще есть, то просто увеличиваем счетчик ссылок
         else:
            self.event_ind = self.event_ind+1
            
        
         self.event_date, self.event_place, self.event_attendence = \
                        self.get_event_inf(self.events_hrefs[self.event_ind]+"?",'i,class,b-list__box-item-title') 

         return self.events_hrefs[self.event_ind]+"?"
     
    
    def get_item_hrefs(self, url_bs4,tag_container,tag,delay):
        tag_list,_ = self.get_items_list_from2tags(url_bs4,tag_container,tag,delay)        
        return [item.attrs['href'] for item in tag_list]
    
    
    def save_class_params(self):
        class_params_init = {}
        class_params_init['cur_url'] = self.cur_url
        class_params_init['cur_page'] = self.params[self.page_param]
        class_params_init['records_pass_in_page_num'] = self.records_pass_in_page_num
        class_params_init['events_url'] = self.events_url
        class_params_init['event_ind'] = self.event_ind
        class_params_init['events_hrefs'] = self.events_hrefs
        class_params_init['event_date'] = self.event_date
        class_params_init['event_place'] = self.event_place
        class_params_init['event_attendence'] = self.event_attendence        

        class_params_init['last_date'] = self.items_list[-1]['Date'] if not len(self.items_list)==0 else ''

        return class_params_init 
    
    
    def load_class_params(self, params):

        # self.items_list = []
        # зачем?
        # self.pause_flag = False
        
        self.cur_url = params['cur_url']
        self.params[self.page_param] = params['cur_page']
        self.records_pass_in_page_num = params['records_pass_in_page_num']
        self.events_url = params['events_url'] 
        self.event_ind = params['event_ind'] 
        self.events_hrefs = params['events_hrefs'] 
        self.event_date = params['event_date']        
        self.event_place = params['event_place'] 
        self.event_attendence = params['event_attendence']

        

    @classmethod    
    def get_head_details(cls, section, tag_container, tag_els, delay):
        sec_head_t,_ = ItemsParser.get_items_list_from2tags(section,tag_container,tag_els, delay)
        sec_head = [item.get_text().strip() for item in sec_head_t]    
        return sec_head
    
    @classmethod              
    def get_fight_det_l(cls,tags_l):
        res_l = [] 
        for item in tags_l:
            l = item.get_text().strip().split('\n')
            res_l.append([item.strip() for i, item in enumerate(l) if i in[0,len(l)-1]])
        return res_l
        
    @classmethod          
    def fill_2fighters_stat(cls,fight_d,keys_l,body_l,round_num=''):
        for i, key in enumerate(keys_l):
            if i!=0:
                fight_d[key+'_l'+round_num] = body_l[i][0]
                fight_d[key+'_r'+round_num] = body_l[i][1]       
    
              
    def get_event_inf(self, url, tag_str):
        html = net_scrape.get_url_delay(delay=self.delay, url = url).text
        bsObj = BeautifulSoup(html, 'lxml')
        tag_d = {'name':tag_str.split(',')[0],'field':tag_str.split(',')[1],\
                  'value':tag_str.split(',')[2]}
        event_inf = []
        event_inf_tags = bsObj.findAll(tag_d['name'],{tag_d['field']:tag_d['value']})
        for i,tag in enumerate(event_inf_tags):
            event_inf.append(tag.parent.get_text().replace(tag.get_text().strip(),'').strip())
    
        return event_inf
                
    def get_one_item_params(self, url):
        
        html = net_scrape.get_url_delay(url = url, delay=self.delay).text
        bsObj = BeautifulSoup(html, 'lxml')
        # словарь с описанием всего события
        fight_desc_d = OrderedDict()

        
        # далее извлекается содержимое
        # описанным ранее методом
        event = bsObj.find('a',{'class':'b-link'}).get_text().strip()
        fight_desc_d['Event'] = event
        
        f_names_t = bsObj.findAll('h3',{'class':'b-fight-details__person-name'})
        f_names = [tag.get_text().strip() for tag in f_names_t]
        fight_desc_d['Fighter_left'] = f_names[0]
        fight_desc_d['Fighter_right'] = f_names[1]
        
        f_res_t = bsObj.findAll('i',{'class':'b-fight-details__person-status'})
        f_res = [tag.get_text().strip() for tag in f_res_t]
        fight_desc_d['Win_lose_left'] = f_res[0]
        fight_desc_d['Win_lose_right'] = f_res[1]
        
        f_det_res_t = bsObj.findAll('i',{'class':'b-fight-details__label'})
       
        det_res = {}
        for i,f_det_res_tag in enumerate(f_det_res_t):
            key = f_det_res_tag.get_text().strip()
            if not key=='Details:':
                det_res[key] = f_det_res_tag.parent.get_text().strip().replace(key,'').strip()
            else:
                det_res[key] = f_det_res_tag.parent.parent.get_text().strip().replace(key,'').strip()
                
        fight_desc_d.update(det_res)
        
        with set_eng_locale():
            event_date = datetime.strptime(self.event_date,'%B %d, %Y')
            
        fight_desc_d['Date'] = event_date
        fight_desc_d['Event_place'] = self.event_place
        fight_desc_d['Event_attendence'] = self.event_attendence  
        
        sections = bsObj.findAll('section', {'class':'b-fight-details__section js-fight-section'})
        sections = [sec for i, sec in enumerate(sections) if not i in [0,3] ] 
        sign_st_parent = bsObj.find('div',{'class':'b-fight-details'})
        for child in sign_st_parent.children:
          try:  
            if  'style' in child.attrs:
                sign_st_table = child
          except:
              pass
        

        sign_act_head = UFCFightsParser.get_head_details(sections[0],'thead,class,b-fight-details__table-head',\
                                                    'th,class,b-fight-details__table-col', self.delay)
        
        
        sign_st_head = UFCFightsParser.get_head_details(sign_st_table,'thead,class,b-fight-details__table-head',\
                                                    'th,class,b-fight-details__table-col', self.delay)
                

                    
        sign_act_body_t,_ = ItemsParser.get_items_list_from2tags(sections[0],'tbody,class,b-fight-details__table-body',\
                                                    'td,class,b-fight-details__table-col', self.delay)
        sign_act_body = UFCFightsParser.get_fight_det_l(sign_act_body_t)
        
        sign_st_body_t,_  = ItemsParser.get_items_list_from2tags(sign_st_table,'tbody,class,b-fight-details__table-body',\
                                                    'td,class,b-fight-details__table-col', self.delay)
        sign_st_body = UFCFightsParser.get_fight_det_l(sign_st_body_t)

            
            
        UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_st_head, sign_st_body)
        UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_act_head, sign_act_body)
        
        for i, key in enumerate(sign_act_head):
            if i!=0:
                fight_desc_d[key+'_l'] = sign_act_body[i][0]
                fight_desc_d[key+'_r'] = sign_act_body[i][1]
        
        
        sign_act_rounds = []
        rounds_body = sections[1].findAll('tr',{'class','b-fight-details__table-row'})
        for i,sec in enumerate(rounds_body):
            if (i!=0):
                sign_act_rounds.append(UFCFightsParser.get_fight_det_l(sec.findAll('td',{'class':'b-fight-details__table-col'})))
    
        for i,round_stat in enumerate(sign_act_rounds):
            UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_act_head,round_stat,round_num=f'_{(i+1)}')
            
    
        sign_st_rounds=[]
        rounds_body = sections[2].findAll('tr',{'class','b-fight-details__table-row'})
        
        for i,sec in enumerate(rounds_body):
            if (i!=0):
                sign_st_rounds.append(UFCFightsParser.get_fight_det_l(sec.findAll('td',{'class':'b-fight-details__table-col'})))
    
        for i,round_stat in enumerate(sign_st_rounds):
            UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_st_head,round_stat,round_num=f'_{(i+1)}')
            
    

        
        return fight_desc_d
    
    
    
   
    
    
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
    url = 'http://www.ufcstats.com/event-details/33b2f68ef95252e0?'
    html = net_scrape.get_url_delay(delay=delay, url = url).text
    bsObj = BeautifulSoup(html, 'lxml')

    event_inf = []
    event_inf_tags = bsObj.findAll('i',{'class':'b-list__box-item-title'})
    for i,tag in enumerate(event_inf_tags):
            event_inf.append(tag.parent.get_text().replace(tag.get_text().strip(),'').strip())
    

    
    # # fight_desc_d_init = {'event':'','fighter_left':'', 'fighter_right':'','win_lose_left':'','win_lose_right':''                         }
    # fight_desc_d = OrderedDict()
    
    # event = bsObj.find('a',{'class':'b-link'}).get_text().strip()
    # fight_desc_d['Event'] = event
    
    # f_names_t = bsObj.findAll('h3',{'class':'b-fight-details__person-name'})
    # f_names = [tag.get_text().strip() for tag in f_names_t]
    # fight_desc_d['Fighter_left'] = f_names[0]
    # fight_desc_d['Fighter_right'] = f_names[1]
    
    # f_res_t = bsObj.findAll('i',{'class':'b-fight-details__person-status'})
    # f_res = [tag.get_text().strip() for tag in f_res_t]
    # fight_desc_d['Win_lose_left'] = f_res[0]
    # fight_desc_d['Win_lose_right'] = f_res[1]
    
    # f_det_res_t = bsObj.findAll('i',{'class':'b-fight-details__label'})
   
    # det_res = {}
    # for i,f_det_res_tag in enumerate(f_det_res_t):
    #     key = f_det_res_tag.get_text().strip()
    #     if not key=='Details:':
    #         det_res[key] = f_det_res_tag.parent.get_text().strip().replace(key,'').strip()
    #     else:
    #         det_res[key] = f_det_res_tag.parent.parent.get_text().strip().replace(key,'').strip()
            
    # fight_desc_d.update(det_res)
    
    # # ------------------------

    
    # sections = bsObj.findAll('section', {'class':'b-fight-details__section js-fight-section'})
    # sections = [sec for i, sec in enumerate(sections) if not i in [0,3] ] 
    # sign_st_parent = bsObj.find('div',{'class':'b-fight-details'})
    # for child in sign_st_parent.children:
    #   try:  
    #     if  'style' in child.attrs:
    #         sign_st_table = child
    #   except:
    #       pass
    

        
    # sign_act_head = UFCFightsParser.get_head_details(sections[0],'thead,class,b-fight-details__table-head',\
    #                                             'th,class,b-fight-details__table-col', delay)
    
    
    # sign_st_head = UFCFightsParser.get_head_details(sign_st_table,'thead,class,b-fight-details__table-head',\
    #                                             'th,class,b-fight-details__table-col', delay)
            


           
    # sign_act_body_t,_ = ItemsParser.get_items_list_from2tags(sections[0],'tbody,class,b-fight-details__table-body',\
    #                                             'td,class,b-fight-details__table-col', delay)
    # sign_act_body = UFCFightsParser.get_fight_det_l(sign_act_body_t)
    
    # sign_st_body_t,_  = ItemsParser.get_items_list_from2tags(sign_st_table,'tbody,class,b-fight-details__table-body',\
    #                                             'td,class,b-fight-details__table-col', delay)
    # sign_st_body = UFCFightsParser.get_fight_det_l(sign_st_body_t)
    

        
    # UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_st_head, sign_st_body)
    # UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_act_head, sign_act_body)
    
    # for i, key in enumerate(sign_act_head):
    #     if i!=0:
    #         fight_desc_d[key+'_l'] = sign_act_body[i][0]
    #         fight_desc_d[key+'_r'] = sign_act_body[i][1]
    
    
    # sign_act_rounds = []
    # rounds_body = sections[1].findAll('tr',{'class','b-fight-details__table-row'})
    # for i,sec in enumerate(rounds_body):
    #     if (i!=0):
    #         sign_act_rounds.append(UFCFightsParser.get_fight_det_l(sec.findAll('td',{'class':'b-fight-details__table-col'})))

    # for i,round_stat in enumerate(sign_act_rounds):
    #     UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_act_head,round_stat,round_num=f'_{(i+1)}')
        

    # sign_st_rounds=[]
    # rounds_body = sections[2].findAll('tr',{'class','b-fight-details__table-row'})
    
    # for i,sec in enumerate(rounds_body):
    #     if (i!=0):
    #         sign_st_rounds.append(UFCFightsParser.get_fight_det_l(sec.findAll('td',{'class':'b-fight-details__table-col'})))

    # for i,round_stat in enumerate(sign_st_rounds):
    #     UFCFightsParser.fill_2fighters_stat(fight_desc_d,sign_st_head,round_stat,round_num=f'_{(i+1)}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # sec_head_t = ItemsParser.get_items_list_from2tags(sections[0],'thead,class,b-fight-details__table-head',\
    #                                             'th,class,b-fight-details__table-col', delay)
    # sec_head = [item.get_text().strip() for item in sec_head_t]
            
    # sign_st_table_head_t = ItemsParser.get_items_list_from2tags(sign_st_table,'thead,class,b-fight-details__table-head',\
    #                                             'th,class,b-fight-details__table-col', delay)        
    # sign_st_table_head = [item.get_text().strip() for item in sign_st_table_head_t] 
    # total_res = bsObj.findAll('th',{'class':'b-fight-details__table-col'})    
    # [item.get_text().strip() for item in total_res]
    
    
    # total_res = ItemsParser.get_items_list_from2tags(bsObj,'tr,class,b-fight-details__table-row',\
    #                                             'th,class,b-fight-details__table-col', delay)
    # [item.get_text().strip() for item in total_res]
    
    
    # total_res = ItemsParser.get_items_list_from2tags(bsObj,'tr,class,b-fight-details__table-row',\
    #                                             'td,class,b-fight-details__table-text', delay)
    
    # fight_desc_tag = bsObj.find('div', {'class':'b-fight-details'})
    # child_l = []
    # put_f = False
    # for i,child in enumerate(fight_desc_tag.children):
    #     if put_f:
    #         child_l.append(child)
    #     if child.find(lambda tag: tag.get_text().strip()=='Totals'):
    #         put_f=True
            

    
    
    # tags  = bsObj.findAll('a',{'style':'color:#B10101'})
    
    
    # ufc_fights.get_next_url(cur_url, tag_container_pages, tag_page)
    
    
    
    
    
    
    
    
    
    
    
    
    