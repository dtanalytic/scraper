
import tkinter as tk
import tkinter.ttk as ttk
import pickle
import os
from datetime import datetime
import configparser
import locale

def construct_page_params(root_win):
    Window(root_win)

class Window(tk.Toplevel):

    def __init__(self, master):
        super().__init__(master)

        self.date = tk.StringVar()
        self.url = tk.StringVar()
        self.cost = tk.StringVar()
        self.page_num_old = tk.IntVar()
        self.num_in_page_old = tk.IntVar()

        date_Entry = tk.Entry(self, textvariable=self.date, width=100)
        date_Label = ttk.Label(self, text='дата')

        cost_Entry = tk.Entry(self, textvariable=self.cost, width=100)
        cost_Label = ttk.Label(self, text='цена')

        url_Entry = tk.Entry(self, textvariable=self.url, width=100)
        url_Label = ttk.Label(self, text='url')

        page_num_old_Entry = tk.Entry(self, textvariable=self.page_num_old, width=100)
        page_num_old_Label = ttk.Label(self, text='№ страницы')

        num_in_page_old_Entry = tk.Entry(self, textvariable=self.num_in_page_old, width=100)
        num_in_page_old_Label = ttk.Label(self, text='№ записи')

        fill_Button = tk.Button(self, text='fill', command=lambda: self.fill_field())
        save_Button=tk.Button(self, text='save', command = lambda: self.saveParams())

        cncl_Button = tk.Button(self, text='cancel', command=self.destroy)


        padWE = dict(sticky=(tk.W, tk.E), padx="0.5m", pady="0.5m")
        date_Entry.grid(row=0, column=0, **padWE)
        date_Label.grid(row=0, column=1, **padWE)

        cost_Entry.grid(row=1, column=0, **padWE)
        cost_Label.grid(row=1, column=1, **padWE)

        url_Entry.grid(row=2, column=0, **padWE)
        url_Label.grid(row=2, column=1, **padWE)

        page_num_old_Entry.grid(row=3, column=0, **padWE)
        page_num_old_Label.grid(row=3, column=1, **padWE)

        num_in_page_old_Entry.grid(row=4, column=0, **padWE)
        num_in_page_old_Label.grid(row=4, column=1, **padWE)

        fill_Button.grid(row=5, column=1, **padWE)

        save_Button.grid(row=6, column=0, **padWE)
        cncl_Button.grid(row=6, column=1, **padWE)

        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=2)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        self.rowconfigure(4, weight=1)
        self.rowconfigure(5, weight=1)
        self.rowconfigure(6, weight=1)

        if os.path.exists('params_page_parser'):
            self.init_fields('params_page_parser')

    def init_fields(self,filename):
        with open(filename,'rb') as f:
            tuple_params = pickle.load(f)

        self.url.set(tuple_params['cur_url'])
        self.num_in_page_old.set(tuple_params['num_records_pass_in_page'])

        self.page_num_old.set(tuple_params['cur_page'])

        self.date.set(tuple_params.get('last_date',0))
        self.cost.set(tuple_params.get('cost', 0))

        #self.cost.set(tuple_params['cost'])

    def fill_field(self):

        # last_date = datetime.today()
        # last_date = last_date.replace(microsecond=0)
        # locale.setlocale(locale.LC_ALL, '')
        # fmt = '%d %B %H:%M %Y'
        # datetime_t = datetime.strftime(last_date,fmt)
        # locale.setlocale(locale.LC_ALL, 'en')
        #
        # self.date.set(datetime_t)
        self.num_in_page_old.set(0)

        self.page_num_old.set(0)

        cfg = configparser.ConfigParser()
        cfg.read('settings.cfg')
        site_name = cfg['general']['site_name']
        self.url.set(cfg[site_name]['cur_url'])




    def saveParams(self):
        tuple_params = {}
        tuple_params['cur_url'] = self.url.get()
        tuple_params['num_records_pass_in_page'] = self.num_in_page_old.get()
        tuple_params['last_date'] = self.date.get()
        tuple_params['cur_page'] = self.page_num_old.get()

        with open('params_page_parser', 'wb') as f:
            pickle.dump(tuple_params, f)



        self.destroy()