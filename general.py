import locale
from contextlib import contextmanager
import os
import configparser

# class ProblGetNextPageException(Exception):
#     pass

class StopException(Exception):
    pass

class NoMoreNewRecordsException(Exception):
    pass

@contextmanager
def set_rus_locale():
    locale.setlocale(locale.LC_ALL, '')
    try:        
        yield
    finally:
        locale.setlocale(locale.LC_ALL, 'en_US.utf8') \
            if os.name == 'posix' else locale.setlocale(locale.LC_ALL, 'en')
        

@contextmanager
def set_eng_locale():
    locale.setlocale(locale.LC_ALL, 'en_US.utf8') \
            if os.name == 'posix' else locale.setlocale(locale.LC_ALL, 'en')
    
    try:        
        yield
    finally:
        locale.setlocale(locale.LC_ALL, '')
                
                
cfg = configparser.ConfigParser()
cfg.read('settings.cfg')
REC_IGN_BEF_STOP_MAX = int(cfg['general']['rec_ign_bef_stop_max'])
PAGES_LOAD_STOP_NUM= int(cfg['general']['pages_load_stop_num'])
