import sys
sys.path.append('../')
from general_modules import net_scrape
from datetime import datetime,timedelta
import unittest
from ufc.ufc_fights_parser import UFCFightsParser
import pickle
import threading
import time
import os
from tkinter import messagebox


class ScraperTest(unittest.TestCase):
    
    # параметры для инициализации объекта в setUp
    true_vals_d={'cur_url':'http://www.ufcstats.com/event-details/ad99fa5325519169?',
                 'events_url':'http://www.ufcstats.com/statistics/events/completed?page=4',
                 
                 }
    # тест на количество боев на странице из true_vals_d['cur_url']
    num_fights = 11
    # тест на следующий url
    next_url = 'http://www.ufcstats.com/event-details/620be7e0712d431b?'
    # проверка сохранения состояния между 
    # исключениями или остановами
    start_stop_vals_d = {
        # останов после стольких записей
        # скачаться успеет на одну больше, так как 
        # пока флаг останова установили очередная запись качается
        'parse_num':12,
        # ссылка на текущий турнир после 25 записей
        'next_url':'http://www.ufcstats.com/event-details/620be7e0712d431b?',
        # номер страницы с турниром на сайте,
        # задается в виде строки для внутрен. манипуляций
        'next_page':'5',
        # номер записи, по которую докачали включительно
        # старт произойдет со следующей
        'records_pass_in_page_num':2,
        # полный url страницы с турнирами после 25 записей
        'next_events_url':'http://www.ufcstats.com/statistics/events/completed?page=5',
        # номер турнира на странице ссылок на турниры
        'event_ind':0
                }
        
    
    def click_pause(self):
        
        while True:
            time.sleep(0.1)            
            if len(self.ufc_fights.items_list)>=int(self.start_stop_vals_d['parse_num']):
                print(len(self.ufc_fights.items_list))
                self.ufc_fights.pause_flag=True
                break
    
    def test_start_stop(self):
        thread = threading.Thread(target=self.click_pause)
        thread.start()
        try:
            self.ufc_fights.start_items_parser()
        except:
            # сохраняем состояние
            with open('params_page_parser', 'wb') as f_w:
                tuple_params = self.ufc_fights.save_class_params()
                pickle.dump(tuple_params, f_w)
                self.assertTrue(os.path.exists('params_page_parser'))
            # получаем параметры состояния
            with open('params_page_parser', 'rb') as f_r:
                saved_d = pickle.load(f_r)
            # проверяем правильность запомненных параметров и истинных значений
            self.assertEqual(self.start_stop_vals_d['next_url'],saved_d['cur_url'])
            self.assertEqual(self.start_stop_vals_d['next_page'],saved_d['cur_page'])
            self.assertEqual(self.start_stop_vals_d['records_pass_in_page_num'],saved_d['records_pass_in_page_num'])
            self.assertEqual(self.start_stop_vals_d['next_events_url'],saved_d['events_url'])
            self.assertEqual(self.start_stop_vals_d['event_ind'],saved_d['event_ind'])

            # инциализируем класс парсера запомненными параметрами
            # и проверяем, что очередная запись будет содержать заданное значение
            self.ufc_fights.load_class_params(saved_d)
            items_hrefs = self.ufc_fights.get_item_hrefs(self.ufc_fights.cur_url,\
                            self.ufc_fights.tag_container_el,self.ufc_fights.tag_el, self.ufc_fights.delay)
            url = items_hrefs[self.ufc_fights.records_pass_in_page_num]
            item_params = self.ufc_fights.get_one_item_params(url)
            self.assertTrue('Montana' in item_params['Fighter_left'])

    
    def setUp(self):
        
        tag_container_events='tbody,,'
        tag_event='a,class,b-link b-link_style_black'
        tag_container_el = 'tbody,class,b-fight-details__table-body'
        tag_el = 'a,class,b-flag b-flag_style_green'
        cur_url = self.true_vals_d['cur_url']
        events_url=self.true_vals_d['events_url']
        page_param = 'page'
        delay = 1
        self.ufc_fights = UFCFightsParser(cur_url,events_url,delay,page_param,\
                                          tag_container_events, tag_event,\
                                          tag_container_el,tag_el)
        

    def test_num_fights(self):
        items_hrefs = self.ufc_fights.get_item_hrefs(self.ufc_fights.cur_url,\
                          self.ufc_fights.tag_container_el,self.ufc_fights.tag_el,\
                          self.ufc_fights.delay)
        self.assertEqual(len(items_hrefs),self.num_fights)

    def test_next_url(self):
        self.assertEqual(self.next_url, self.ufc_fights.get_next_url())
        



if __name__=='__main__':
    
    # import tkinter as tk
    
    # root = tk.Tk()
    # messagebox.showinfo('остановка')
    # root.mainloop()
    # unittest.main()
    import pandas as pd 
    df = pd.read_csv('items.csv')
    
    # # df2 = df[['Date','event_place','Event']]
        
    # with open('params_page_parser','rb') as f_r:
    #     d = pickle.load(f_r)
    
    # with open('ark/params_page_parser','rb') as f_r:
    #     d2 = pickle.load(f_r)
    
    # df2 = pd.read_csv('ark/items.csv')

    # df.Event.nunique()
    # df.loc[df.Event.str.contains('UFC \d{1,}'),'Event' ].nunique()   
