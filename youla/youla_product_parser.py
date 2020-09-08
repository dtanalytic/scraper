import shop_product_parser
import json
from bs4 import BeautifulSoup
import time
import sys
sys.path.append('../')
from general_modules import net_scrape


class YoulaProductParser(shop_product_parser.ShopProductsParser):

    PRODUCT_ITEM_HREF_CLASS = 'href'

    def __init__(self,cur_url, tag_container_el,tag_el,page_param, url_alt,pages_load_stop):
        super().__init__(cur_url, tag_container_el,tag_el,page_param,pages_load_stop, url_alt)


    def getNextURL(self, url, tag_container, tag):

        self.params[self.page_param] = str(int(self.params[self.page_param]) + 1)
        next_page=self.makeUrlFromParts(self.url_base, self.url_add, self.params)
        #response = self.afterConnectionDelay(next_page)
        response = net_scrape.get_url_delay(self.delay, next_page)
        json_dict = json.loads(response.content)
        if 'error' in json_dict.keys():return ''
        else: return next_page

    def getTimeProduct(self, str):

       import locale
       locale.setlocale(locale.LC_ALL, '')
       #fmt_to_datetime = '%a %b %d %H:%M:%S %Y'
       #a = time.strptime(datetime_str, fmt_to_datetime)
       fmt_from_datetime = '%d %B %H:%M %Y'
       datetime = time.strftime(fmt_from_datetime, time.localtime(str))
       # locale.setlocale(locale.LC_ALL, 'en')
       locale.setlocale(locale.LC_ALL, 'en_US.utf8')
       return datetime

    def getProductParams(self, product_params, url):

        #html = self.afterConnectionDelay(url)
        html = net_scrape.get_url_delay(self.delay, url)
        json_dict = json.loads(html.content)

        data = json_dict['data']

        for key,value in self.product_tags.items():
            if type(key) is str and key in data.keys():
                product_params[value] = data[key]
            elif key[0]=='attributes':
                if len([x for x in data['attributes'] if x['slug'] == key[1]]) == 1:
                    product_params[value] = [x for x in data['attributes'] if x['slug'] == key[1]][0]['values'][0]['value']

            else:
                product_params[value]=data[key[0]][key[1]]
        product_params['page_num'] = self.params[self.page_param]



    def getListRefJson(self,cur_url, tag_el):
        #response = self.afterConnectionDelay(cur_url)
        response = net_scrape.get_url_delay(self.delay, cur_url)
        #data = json.loads(response)
        data = json.loads(response.content)
        tags = data['html']

        bsObj = BeautifulSoup(tags, 'lxml')
       # tag_name = 'li'
       # class_name = 'product_item'

        #tag_list = bsObj.find_all(tag_name, class_=class_name)
        tag_list = bsObj.find_all(tag_el['name'], class_=tag_el['class'])
        new_tag_list = [tag.contents[0].attrs['href'] for tag in tag_list]

        return new_tag_list