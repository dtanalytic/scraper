
from general_modules import net_scrape
from bs4 import BeautifulSoup

delay = 1
UFC_FIGHT_TAGS = [('a','class','b-link b-link_style_black')
                  ]


# def get_scrape_items_list(url,tag_container_list):
#         items_list = []
#         html = net_scrape.get_url_delay(delay, url).text
#         bsObj = BeautifulSoup(html, 'lxml')
#         try:
#             items_list = bsObj.find(tag_container['name'],class_=tag_container['class']).find_all(tag['name'],class_=tag['class'])
#         finally:
#             return items_list
        
        
            
            
    

if __name__=='__main__':
    delay = 1
    url = 'http://ufcstats.com/statistics/events/completed'
    html = net_scrape.get_url_delay(delay=delay, url = url).text
    bsObj = BeautifulSoup(html, 'lxml')
    tag_container_list = []    
    
    hrefs = bsObj.find_all('a',{'class':'b-link b-link_style_black'})
    hrefs = [item.attrs['href'] for item in hrefs]