
import shop_product_parser
import re
from datetime import date, timedelta, datetime
import time


class problGetNextPageException(Exception):
    pass

class AvitoProductParser(shop_product_parser.ShopProductsParser):


    #PRODUCT_ITEM_HREF_CLASS = 'item-description-title-link'

    PRODUCT_ITEM_HREF_CLASS = 'snippet-link'
    def __init__(self,cur_url, tag_container_el,tag_el,tag_container_pages, tag_page, page_param, pages_load_stop):

        self.tag_container_pages = tag_container_pages if isinstance(tag_container_pages, dict) else {
            'name': tag_container_pages.split(',')[0], 'class': tag_container_pages.split(',')[1]}
        self.tag_page = tag_page if isinstance(tag_page, dict) else {'name': tag_page.split(',')[0],
                                                                     'class': tag_page.split(',')[1]}
        super().__init__(cur_url, tag_container_el,tag_el,page_param, pages_load_stop)


    # def getNextURL(self,url,tag_container,tag):
    #     try:
    #         list_tags = self.getListFrom2Tags(url, tag_container, tag)
    #         if list_tags:
    #             total_pages_piece = list_tags[-1].get('href')
    #             url_base, url_add, params = self.parseUrl(self.url_base + total_pages_piece)
    #             pages_count = params[self.page_param]
    #             current_page_num= self.params[self.page_param]
    #             if int(current_page_num)+1<=int(pages_count):
    #                 self.params[self.page_param] = str(int(current_page_num)+1)
    #                 return self.makeUrlFromParts(self.url_base, self.url_add, self.params)
    #             else:
    #                 return ''
    #
    #         else:
    #             raise problGetNextPageException
    #     finally:
    #         pass
    #         #print('problems with getting next page')
    #     # except problGetNextPageException as e:
    #     #     print('problems with getting next page or they are out')
    #     #     return ''
    #
    #     # заполняет словарь параметров для объекта
    def getNextURL(self,url,tag_container,tag):

            list_tags = self.getListFrom2Tags(url, tag_container, tag)
            if list_tags:
                pages_count = list_tags[-2].text
                url_base, url_add, params = self.parseUrl(url)

                current_page_num= self.params[self.page_param]
                if int(current_page_num)+1<=int(pages_count):
                    self.params[self.page_param] = str(int(current_page_num)+1)
                    return self.makeUrlFromParts(self.url_base, self.url_add, self.params)



            else:
                 raise problGetNextPageException




    def getProductParams(self, product_params, bsObj):

            for (tag, var) in self.product_tags.items():
                try:
                    if tag[1] == 'text':
                        span_str = bsObj.find(text=re.compile(tag[2])).parent.get_text()
                        whole_str = bsObj.find(text=re.compile(tag[2])).parent.parent.get_text()
                        product_params[var] = whole_str.replace(span_str, '')

                    else:
                        product_params[var] = bsObj.find(tag[0], {tag[1]: tag[2]}).get_text()
                    product_params[var] = product_params[var].strip()
                except:
                    pass


    def getTimeProduct(self, str1):
        import locale
        locale.setlocale(locale.LC_ALL, '')
        re_parts = re.search('(сегодня|вчера|\d+\s+[А-Яа-я]+)\s{0,}в{0,}\s{0,}(\d+:\d+)',str1)
        date_time = ''
        if re_parts:
            date_part, time_part=re_parts.groups()


            if date_part=='сегодня':
                    date_time = date_time + date.today().strftime('%d %B')
            elif date_part=='вчера':
                    yesterday = date.today() - timedelta(days=1)
                    date_time = date_time + yesterday.strftime('%d %B')
            else:
                    date_time = date_part

            if not date_time.lower().find('декабрь')==-1 and date.today().month==1:
                date_time = date_time + ' ' + time_part + ' ' + str(datetime.now().year-1)
            else:
                date_time = date_time +' ' +time_part+ ' ' + str(datetime.now().year)


        locale.setlocale(locale.LC_ALL, 'en')
        return date_time

