import youla.youla_product_parser
import queue
import json



class YoulaFlatParser(youla.youla_product_parser.YoulaProductParser):

    product_tags = {'price': 'cost',
                    ('attributes', 'realty_obshaya_ploshad'): 'total_square',
                    ('attributes', 'komnat_v_kvartire'): 'rooms_num',
                    ('attributes', 'realty_etaj'): 'floor',
                    ('attributes', 'realty_etajnost_doma'): 'total_floors',
                    ('attributes', 'tip_doma'): 'house_type',
                    ('attributes', 'subway_station'): 'metro',
                    ('attributes', 'realty_god_postroyki'): 'build_date',
                    ('attributes', 'realty_ploshad_kuhni'): 'live_square',
                    ('attributes', 'rasstoyanie_ot_metro'): 'm_distance',
                    ('attributes', 'subcategory'): 'subcategory',
                    ('location','description'): 'adr',
                    'date_published': 'date_time',
                    'description': 'desc'  # проверить, возможно меняется
                    }


    product_params_init = {'cost': '', 'total_square': '', 'live_square': '', 'rooms_num': '', 'floor': '',
                           'total_floors': '', 'metro': '', 'm_distance': '', 'adr': '', 'date_time': '',
                           'desc': '', 'house_type': '', 'page_num': '', 'build_date': '', 'subcategory':''}

    sub_category_value = 'Продажа квартиры'
    delay = 1 / 4

    def __init__(self,cur_url, tag_container_el,tag_el,page_param, url_alt,pages_load_stop):
        super().__init__(cur_url, tag_container_el,tag_el,page_param, url_alt,pages_load_stop)

    # def getWholeProductParams1(self, product_params, html):
    #     # нужная ссылка сюда должна попасть с учетом url_alt
    #     json_dict = json.loads(html.content)
    #     data = json_dict['data']
    #     list = data['attributes']
    #
    #     product_params['cost'] = data['price']
    #     product_params['desc'] = data['description']
    #     product_params['date_time'] = data['date_published']
    #     product_params['adr'] = data['location']['description']
    #
    #     if len([x for x in list if x['slug'] == "komnat_v_kvartire"]) == 1:
    #         product_params['rooms_num'] = [x for x in list if x['slug'] == "komnat_v_kvartire"][0]['values'][0]['value']
    #
    #     if len([x for x in list if x['slug'] == "realty_obshaya_ploshad"]) == 1:
    #         product_params['total_square'] = [x for x in list if x['slug'] == "realty_obshaya_ploshad"][0]['values'][0][
    #                                              'value'] / 10
    #
    #     if len([x for x in list if x['slug'] == "realty_etaj"]) == 1:
    #         product_params['floor'] = [x for x in list if x['slug'] == "realty_etaj"][0]['values'][0]['value']
    #
    #     if len([x for x in list if x['slug'] == "tip_doma"]) == 1:
    #         product_params['house_type'] = [x for x in list if x['slug'] == "tip_doma"][0]['values'][0]['value']
    #
    #     if len([x for x in list if x['slug'] == "subway_station"]) == 1:
    #         product_params['metro'] = [x for x in list if x['slug'] == "subway_station"][0]['values'][0]['value']
    #
    #     if len([x for x in list if x['slug'] == "realty_ploshad_kuhni"]) == 1:
    #         product_params['live_square'] = product_params['total_square'] - \
    #                                         [x for x in list if x['slug'] == "realty_ploshad_kuhni"][0]['values'][0][
    #                                             'value'] / 10
    #
    #     if len([x for x in list if x['slug'] == "realty_etajnost_doma"]) == 1:
    #         product_params['total_floors'] = [x for x in list if x['slug'] == "realty_etajnost_doma"][0]['values'][0][
    #             'value']
    #
    #     if len([x for x in list if x['slug'] == "realty_god_postroyki"]) == 1:
    #         product_params['build_date'] = [x for x in list if x['slug'] == "realty_god_postroyki"][0]['values'][0][
    #             'value']
    #
    #     if len([x for x in list if x['slug'] == "rasstoyanie_ot_metro"]) == 1:
    #         product_params['m_distance'] = [x for x in list if x['slug'] == "rasstoyanie_ot_metro"][0]['values'][0][
    #             'value']
    #
    #     product_params['page_num'] = self.params[self.page_param]

    def getWholeProductParams(self, product_params, url):

        self.getProductParams(product_params, url)

        product_params['date_time']=self.getTimeProduct(product_params['date_time'])
        product_params['cost'] =product_params['cost'] /100

        if not product_params['total_square']=='':
            product_params['total_square'] =product_params['total_square']/10
        if not product_params['live_square']=='':
            product_params['live_square'] = product_params['total_square'] - product_params['live_square']/10



    def filterProductAcToSite(self,product_params):
        if not product_params['subcategory'] == self.sub_category_value or product_params['cost'] == '':
            del product_params['subcategory']
            return False
        else:

            del product_params['subcategory']
            return True
    # def getWholeProductParams(product_tags,product_params, html):
    #     json_dict = json.loads(html.content)
    #     data = json_dict['data']
    #
    #     for key,value in product_tags.items():
    #         if type(key) is str:
    #             product_params[value] = data[key]
    #         elif key[0]=='attributes':
    #             if len([x for x in data['attributes'] if x['slug'] == key[1]]) == 1:
    #                 product_params[value] = [x for x in data['attributes'] if x['slug'] == key[1]][0]['values'][0]['value']
    #         else:
    #             product_params[value]=data[key[0]][key[1]]


#debug(getWholeProductParams,product_tags,product_params, response)


if __name__=='__main__':
    # параметры нужно задать как словарь так как порядок не соблюден

    from bs4 import BeautifulSoup
    import json
    from selenium import webdriver
    import time
    import re



    a = YoulaFlatParser('https://www.youla.ru/web-api/products?cId=20&city=576d0612d53f3d80945f8b5d&page=1&serpId=36f5ec8f7cc', 'ul,product_list _board_items','li,product_item','page','',10)
    #messages_queue = queue.Queue()
    #a.startListProductsParser(messages_queue)
    product_params = {'cost': '', 'total_square': '', 'live_square': '', 'rooms_num': '', 'floor': '',
                           'total_floors': '', 'metro': '', 'm_distance': '', 'adr': '', 'date_time': '',
                           'desc': '', 'house_type': '', 'page_num': '', 'build_date': ''}



    a.getProductParams(product_params, 'https://api.youla.io/api/v1/product/5c0f527262e1c686fb437bb2')

    datetime_str = time.ctime(product_params['date_time'])
    import locale

    locale.setlocale(locale.LC_ALL, '')

    # fmt_to_datetime = '%a %b %d %H:%M:%S %Y'
    # a = time.strptime(datetime_str, fmt_to_datetime)

    fmt_from_datetime = '%d %B %H:%M %Y'
    datetime = time.strftime(fmt_from_datetime,time.localtime(product_params['date_time']))
    locale.setlocale(locale.LC_ALL, 'en')