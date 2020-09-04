
from avito import avito_product_parser
from bs4 import BeautifulSoup
import re
import numpy as np
import os
import pandas as pd
import requests
import json
from geopy import distance
import sys
sys.path.append('../')
from general_modules import net_scrape
import time
import ctypes

def set_trace():
	from IPython.core.debugger import Pdb
	import sys
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

class AvitoFlatParser(avito_product_parser.AvitoProductParser):


    product_tags = {('span', 'class', 'js-item-price'): 'cost',
                 ('span', 'text', 'Общая площадь:'): 'total_square',
                 ('span', 'text', 'Жилая площадь:'): 'live_square',
                 ('span', 'text', 'Количество комнат:'): 'rooms_num',
                 ('span', 'text', 'Этаж:'): 'floor',
                 ('span', 'text', 'Этажей в доме:'): 'total_floors',
                 ('span', 'text', 'Тип дома:'): 'house_type',
                 ('span', 'text', 'Отделка:'): 'rep_type',
                 ('span', 'text', 'Официальный застройщик:'): 'builder',
                 # ('span', 'item-address__string', 'address'): 'adr',
                 # ('span', 'item - address - georeferences - item__content', 'district'): 'district',
                 ('span', 'class', 'item-address__string'): 'adr',
                 ('span', 'class', 'item-address-georeferences-item__content'): 'district',
                 ('div', 'class', 'title-info-metadata-item-redesign'): 'date_time',
                 ('div', 'class', 'item-description'): 'desc'  # проверить, возможно меняется
                 }

    # удалить если не понадобится
    product_params_init = {'cost': '', 'total_square': '', 'live_square': '', 'rooms_num': '', 'floor': '','rep_type':'',
                           'total_floors': '', 'metro': '', 'm_distance': '', 'adr': '', 'date_time': '','builder':'',
                           'desc': '', 'house_type': '', 'page_num': '','dist_cent':'','build_age':'','district':'','lat':'','lon':''}

    delay = 5
    flats_age_filename = r'c:\work\dev\python\progs\scraper\flats_age.csv'
    city_name ='Владикавказ'
    city_cent =(43.024531, 44.682651)

    flag_continue_searching_dist=True

    def __init__(self,cur_url, tag_container_el,tag_el,tag_container_pages, tag_page, page_param, pages_load_stop):
        super().__init__(cur_url, tag_container_el,tag_el,tag_container_pages, tag_page, page_param,pages_load_stop)

    @classmethod
    def makeStbsFloat(cls,df,list_stbs):
        for item in list_stbs:
            if df[item].dtype == np.object or df[item].dtype == np.string_:
                df[item] = df[item].apply(lambda x: x.replace(' ', '') if pd.notnull(x) else x)
                df[item] = df[item].astype(np.float64)

    # @classmethod
    # def get_house_age(cls,df_flat_age):

    #     df_flat_age['улица']=df_flat_age['улица'].str.replace('пер\.{0,1}','')
    #     df_flat_age['улица']=df_flat_age['улица'].str.replace('/.*','')
    #     df_flat_age['улица']=df_flat_age['улица'].str.replace('\(.*','')
    #     df_flat_age['улица']=df_flat_age['улица'].str.replace('пр.','')
    #     df_flat_age['улица']=df_flat_age['улица'].str.replace('Проспект','')
    #     df_flat_age['улица'] = df_flat_age['улица'].str.strip()

    #     df_flat_age['дом']=df_flat_age['дом'].str.strip()
    #     df_flat_age['cl_adr']=df_flat_age['улица']+';'+df_flat_age['дом']

    #     df_flat_age=df_flat_age.drop(df_flat_age[df_flat_age['год постройки'].isnull()].index)
    #     df_flat_age=df_flat_age.drop(df_flat_age[df_flat_age['cl_adr'].isnull()].index)

    #     df_flat_age=df_flat_age[['год постройки','cl_adr']]

    #     df_flat_age.loc[df_flat_age['cl_adr'].notnull(), 'cl_adr'] = df_flat_age.loc[df_flat_age['cl_adr'].notnull(), 'cl_adr'].map(lambda x: x.strip())

    #     df_flat_age = df_flat_age.set_index(['cl_adr'])

    #     return df_flat_age.T.to_dict('list')
    
    @classmethod
    def get_house_param(cls,df_flat_age,param):

        df_flat_age['улица']=df_flat_age['улица'].str.replace('пер\.{0,1}','')
        df_flat_age['улица']=df_flat_age['улица'].str.replace('/.*','')
        df_flat_age['улица']=df_flat_age['улица'].str.replace('\(.*','')
        df_flat_age['улица']=df_flat_age['улица'].str.replace('пр.','')
        df_flat_age['улица']=df_flat_age['улица'].str.replace('Проспект','')
        df_flat_age['улица'] = df_flat_age['улица'].str.strip()

        df_flat_age['дом']=df_flat_age['дом'].str.strip()
        df_flat_age['cl_adr']=df_flat_age['улица']+';'+df_flat_age['дом']

        df_flat_age=df_flat_age.drop(df_flat_age[df_flat_age[param].isnull()].index)
        df_flat_age=df_flat_age.drop(df_flat_age[df_flat_age['cl_adr'].isnull()].index)

        df_flat_age=df_flat_age[[param,'cl_adr']]

        df_flat_age.loc[df_flat_age['cl_adr'].notnull(), 'cl_adr'] = df_flat_age.loc[df_flat_age['cl_adr'].notnull(), 'cl_adr'].map(lambda x: x.strip())

        df_flat_age = df_flat_age.set_index(['cl_adr'])

        return df_flat_age.T.to_dict('list')
    
    @classmethod
    def dist_leven(cls,a, b):
        "Calculates the Levenshtein distance between a and b."
        n, m = len(a), len(b)
        if n > m:
            # Make sure n <= m, to use O(min(n,m)) space
            a, b = b, a
            n, m = m, n

        current_row = range(n + 1)  # Keep current and previous row, not entire matrix
        for i in range(1, m + 1):
            previous_row, current_row = current_row, [i] + [0] * n
            for j in range(1, n + 1):
                add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
                if a[j - 1] != b[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        return current_row[n]


    @classmethod
    def levenDistForNull(cls,str, dict_house_age):
        try:
            #print('столбец таблицы:' + str +'\n')
            for key, value in dict_house_age.items():
                if AvitoFlatParser.dist_leven((key.split(';')[0]).lower(), (str.split(';')[0]).lower()) <= 1 and key.split(';')[1].lower()==str.split(';')[1].lower():
                    #set_trace()
                    #print ('столбец таблицы:'+str+'\tстолбец словаря:'+key+'\n')
                    return float(value[0])

        except Exception as e:

            return None


    @classmethod
    def calc_flats_age(cls,df, start, end, dict_house_age):
        def map_dict_house_age(str):
            #print('строка:'+str[0]+'\n')
            try:
                return float(str[0].replace(' ', ''))
            except:pass

        stbs_part = df.loc[range(start, end), ['build_age', 'cl_adr']]
        stbs_part['build_age'] = stbs_part['cl_adr'].map(dict_house_age)

        stbs_part.loc[stbs_part['build_age'].notnull(), 'build_age'] = stbs_part.loc[
            stbs_part['build_age'].notnull(), 'build_age'].map(map_dict_house_age)

        # stbs_part.loc[stbs_part['build_age'].isnull(), 'build_age'] = stbs_part.loc[ \
        #     stbs_part['build_age'].isnull(), 'cl_adr'].map( \
        #     lambda x: AvitoFlatParser.levenDistForNull(x, dict_house_age))

        df.loc[range(start, end), ['build_age', 'cl_adr']] = stbs_part





    @classmethod
    def streetNumber(cls,str):

        street = ''
        street_part = ''
        num = ''
        one_symb = ''

        re_parts = re.search('(ул|пер|пр-т|пр-кт|пр|Ул|Пер|Пр)(\.|ица|еулок|оспект){0,1}([^а-я])([А-Яа-я\s\.]{0,}).*?(\d{1,4})', str)


        if re_parts:
            __, __, one_symb, street, num = re_parts.groups()
            street_part = street.strip()
            street = one_symb + street_part
            street = street.strip()
            num = num.strip()

        if street_part == '' or num == '':
            re_parts = re.search(',*([А-Яа-я\s\.]{0,})(ул|пер|пр-т|пр-кт|пр|Ул|Пер|Пр)(\.|ица|еулок|оспект){0,1}[,\s]*(\d{1,4})', str)
            if re_parts:
                street, __, __, num = re_parts.groups()
                street = street.strip()
                num = num.strip()

        return street + ';' + num



    @classmethod
    def stbDistCent(cls, str,city_name, city_cent):
        dist = ''
        if re.search(r'[а-я\d]{4,}',str):
            street, num = str.split(';')


            try:
                set_trace()
                data = requests.get(
                    'https://geocode-maps.yandex.ru/1.x/?geocode=' + city_name + ',' + street + ',' + num + '&format=json').text
                data = json.loads(data)
                if data['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point']['pos']:
                    lon, lat = data['response']['GeoObjectCollection']['featureMember'][0]['GeoObject']['Point'][
                        'pos'].split(
                        ' ')


                    dist = distance.distance(city_cent, (lat, lon)).km
                    time.sleep(1)

            except Exception as e:
                if cls.flag_continue_searching_dist==True:
                    result = ctypes.windll.user32.MessageBoxW(0,
                                                              "Качать адреса изменив ip \n(или продолжить без скачивания)",
                                                              "Warning!", 4)
                    # нажато да
                    if result == 6:
                        print("hope you changed ip address")
                    # нажато нет
                    elif result == 7:
                        cls.flag_continue_searching_dist=False
                        raise

                else:
                    print(str)

        return dist

    @classmethod
    def calc_distance_center(cls, df,first,last,city_name, city_cent):

            #df.loc[range(first, last),'cl_adr']=None
            #df.loc[range(first,last),'cl_adr']
            # то же самое df['cl_adr']=df['adr'].map(AvitoFlatParser.streetNumber)
            
            # перенес в df_tester
            # df.loc[range(first, last), 'cl_adr']=df.loc[range(first, last), 'adr'].map(AvitoFlatParser.streetNumber,str)
            df.loc[range(first, last), 'dist_cent'] = df.loc[range(first, last), 'cl_adr'].apply(AvitoFlatParser.stbDistCent, args=(city_name, city_cent))




    @classmethod
    def cleanPrepare(cls,df,list_stbs_float):

        df.loc[df['total_square'].notnull(), 'total_square'] = df.loc[df['total_square'].notnull(), 'total_square'].map(
            lambda x: x.replace('м²', ''))
        df.loc[df['live_square'].notnull(), 'live_square'] = df.loc[df['live_square'].notnull(), 'live_square'].map(
            lambda x: x.replace('м²', ''))
        df['rooms_num'] = np.where(df.rooms_num.str.contains('студии'), '1', df.rooms_num.str.findall('(\d+)').str[0])

        AvitoFlatParser.makeStbsFloat(df,list_stbs_float)

    def getWholeProductParams(self, product_params, url):
        #html = self.afterConnectionDelay(url).text
        html =  net_scrape.get_url_delay(self.delay,url).text

        bsObj = BeautifulSoup(html, 'lxml')
        self.getProductParams(product_params, bsObj)
        metro_text=''
        list_metros=bsObj.findAll('span', {'class': 'item-map-metro'})
        for metro in list_metros:
            metro_text=metro_text+metro.get_text()

        product_params['metro'] = metro_text.strip().replace('\n','').replace('  ', ' ')
        list_dist = re.findall('\(\d+[\d,.]*\s[км]{1,2}\)',product_params['metro'])
        product_params['m_distance'] = self.returnMinMetroDistance(list_dist)
        product_params['date_time'] = self.getTimeProduct(product_params['date_time'])
        product_params['lat'],product_params['lon'] = self.get_lat_lon(bsObj)
        product_params['page_num'] = self.params[self.page_param]

    def get_lat_lon(self,bsObj):
        
        div_str = bsObj.find('div',{'class':'b-search-map expanded item-map-wrapper js-item-map-wrapper'})
        return div_str.attrs['data-map-lat'], div_str.attrs['data-map-lon']
        
        
    def returnMinMetroDistance(self, list_dist):
        if len(list_dist) == 0:
            return ''
        else:
            try:
                list_dist = [item.replace(',','.') for item in list_dist]
                clean_list1 = [(item.split()[0], item.split()[1]) for item in list_dist]
                clean_list2 = [(num[1:], meas[:-1]) for (num, meas) in clean_list1]
                clean_list3 = np.array(clean_list2)
                dist = clean_list3[:,0].astype(np.float64)

                dist[clean_list3[:,1]=='км']= dist[clean_list3[:,1]=='км']*1000

                return str(dist.min())

            except: return ''

    def filterProductAcToSite(self, product_params):
        if product_params['cost']=='':
            return False
        else: return True


if __name__=='__main__':
    
        html =  net_scrape.get_url_delay(1,'http://ufcstats.com/statistics/events/completed').text

        bsObj = BeautifulSoup(html, 'lxml')
        fights = bsObj.findAll('tr',{'class':'b-statistics__table-row'})
        
        hrefs = bsObj.findAll('a',{'class':'b-link b-link_style_black'})
        
        hrefs = [item.attrs['href'] for item in hrefs]
        
            # for (tag, var) in self.product_tags.items():
            #     try:
            #         if tag[1] == 'text':
            #             span_str = bsObj.find(text=re.compile(tag[2])).parent.get_text()
            #             whole_str = bsObj.find(text=re.compile(tag[2])).parent.parent.get_text()
            #             product_params[var] = whole_str.replace(span_str, '')

            #         else:
            #             product_params[var] = bsObj.find(tag[0], {tag[1]: tag[2]}).get_text()
            #         product_params[var] = product_params[var].strip()
            #     except:
            #         pass
    

        # product_tags = {('span', 'class', 'js-item-price'): 'cost',
        #          ('span', 'text', 'Общая площадь:'): 'total_square',
        #          ('span', 'text', 'Жилая площадь:'): 'live_square',
        #          ('span', 'text', 'Количество комнат:'): 'rooms_num',
        #          ('span', 'text', 'Этаж:'): 'floor',
        #          ('span', 'text', 'Этажей в доме:'): 'total_floors',
        #          ('span', 'text', 'Тип дома:'): 'house_type',
        #          ('span', 'text', 'Отделка:'): 'rep_type',
        #          ('span', 'text', 'Официальный застройщик:'): 'builder',
        #          ('span', 'class', 'item-address__string'): 'adr',
        #          ('span', 'class', 'item-address-georeferences-item__content'): 'district',
        #          ('div', 'class', 'title-info-metadata-item-redesign'): 'date_time',
        #          ('div', 'class', 'item-description'): 'desc'  # проверить, возможно меняется
        #          }

        # product_params_init = {'cost': '', 'total_square': '', 'live_square': '', 'rooms_num': '', 'floor': '','rep_type':'',
        #                    'total_floors': '', 'metro': '', 'm_distance': '', 'adr': '', 'date_time': '','builder':'',
        #                    'desc': '', 'house_type': '', 'page_num': '','dist_cent':'','build_age':'','district':''}

