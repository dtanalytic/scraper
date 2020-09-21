import sys
sys.path.append('../')
from general_modules import net_scrape
from datetime import datetime,timedelta
import unittest
from ufc.ufc_fights_parser import UFCFightsParser

class ScraperTest(unittest.TestCase):
    
    def setUp(self):
        cur_url = 'http://www.ufcstats.com/event-details/542db012217ecb83?'
        events_url='http://www.ufcstats.com/statistics/events/completed?page=1'
        tag_container_events='tbody,,'
        tag_event='a,class,b-link b-link_style_black'
        tag_container_el = 'tbody,class,b-fight-details__table-body'
        tag_el = 'a,class,b-flag b-flag_style_green'
        page_param = 'page'
        delay = 1
        self.ufc_fights = UFCFightsParser(cur_url,events_url,delay,page_param,tag_container_events, tag_event, tag_container_el,tag_el)
        

    def test_step_page(self):
        # ufc_fights.start_items_parser()
        hrefs=['http://ufcstats.com/fight-details/f8dd1e75978a3957',
               'http://ufcstats.com/fight-details/d395828f5cb045a5'
     
               ]
        
        item_last_datetime = datetime.now()-timedelta(days=30)
        items = self.ufc_fights.get_items_params(hrefs, item_last_datetime = item_last_datetime)
        self.assertEqual(len(items),3)

        
if __name__=='__main__':
    unittest.main(warnings='ignore')