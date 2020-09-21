import tkinter as tk

import configparser
from avito import avito_flat_parser
from youla import youla_flat_parser
from ufc import ufc_fights_parser


class Window(tk.Toplevel):
    site_name=''

    def __init__(self,master):
        super().__init__(master)
        self.create_variables()
        self.create_widgets()
        self.create_layout()
        self.focus_set()    # принять фокус ввода,
        self.grab_set()     # запретить доступ к др. окнам, пока открыт диалог
        self.wait_window()

    @classmethod
    def get_site_class(cls):

        if cls.site_name=='':
            cfg = configparser.ConfigParser()
            cfg.read('settings.cfg')
            cls.site_name = cfg['general']['site_name']

        if cls.site_name=='avito':return avito_flat_parser.AvitoFlatParser(**Window.get_site_params('avito'))

        if cls.site_name=='youla':return youla_flat_parser.YoulaFlatParser(**Window.get_site_params('youla'))
        
        if cls.site_name=='ufc_fights':return ufc_fights_parser.UFCFightsParser(**Window.get_site_params('ufc_fights'))



    @classmethod
    def get_site_params(clk,site):
        cfg = configparser.ConfigParser()
        cfg.read('settings.cfg')
        init_params_dict = {i:j for i,j in cfg[site].items()}
        init_params_dict['pages_load_stop_num'] = int(cfg['general']['pages_load_stop_num'])
        init_params_dict['rec_ign_bef_stop_max'] = int(cfg['general']['rec_ign_bef_stop_max'])
        init_params_dict['delay'] = int(cfg['general']['delay'])
        return init_params_dict

    def saveParams(self):
        init_params_dict = {}
        init_params_dict['tag_container_el'] = self.rec_cont.get()
        init_params_dict['tag_el'] = self.rec_cont_inside.get()
        init_params_dict['tag_container_pages'] = self.pag_cont.get()
        init_params_dict['tag_page'] = self.pag_cont_inside.get()
        init_params_dict['page_param'] = self.page_param.get()
        init_params_dict['cur_url'] = self.cur_url.get()

        init_params_dict['pages_load_stop'] = self.pages_load_stop.get()

        #провеоить работу сменщика сайта
        Window.site_name= self.site_radio_but.get()
        self.master.page_parser = Window.get_site_class()
        #self.master.page_parser = avito_flat_parser.AvitoFlatParser(**init_params_dict)

        self.destroy()

    def onPressRadioBut(self):
        site = self.site_radio_but.get()
        init_params_dict = Window.get_site_params(site)
        self.setSiteParams(init_params_dict)

    def setSiteParams(self, init_params_dict):
        self.rec_cont.set(init_params_dict['tag_container_el'])
        self.rec_cont_inside.set(init_params_dict['tag_el'])
        self.pag_cont.set(init_params_dict['tag_container_pages'])
        self.pag_cont_inside.set(init_params_dict['tag_page'])
        self.page_param.set(init_params_dict['page_param'])
        self.cur_url.set(init_params_dict['cur_url'])

        self.pages_load_stop.set(init_params_dict['pages_load_stop'])

    def create_variables(self):

        self.rec_cont = tk.StringVar()
        self.rec_cont_inside = tk.StringVar()
        self.pag_cont = tk.StringVar()
        self.pag_cont_inside = tk.StringVar()
        self.page_param = tk.StringVar()
        self.cur_url = tk.StringVar()
        self.site_radio_but = tk.StringVar()
        self.pages_load_stop = tk.StringVar()


    def create_widgets(self):

        self.start_url_Entry = tk.Entry(self,textvariable=self.cur_url, width=100)

        self.container_el_Entry=tk.Entry(self,width=30,textvariable=self.rec_cont)
        self.el_Entry = tk.Entry(self, width=30,textvariable=self.rec_cont_inside)

        self.container_pages_Entry = tk.Entry(self, width=30,textvariable=self.pag_cont)
        self.pages_Entry = tk.Entry(self, width=30,textvariable=self.pag_cont_inside)

        self.page_param_Entry = tk.Entry(self, width=2,textvariable=self.page_param)

        self.pages_load_stop_Entry = tk.Entry(self, width=4,textvariable=self.pages_load_stop)

        self.save_Button=tk.Button(self, text='save', command = lambda: self.saveParams())

        self.cncl_Button = tk.Button(self, text='cancel', command=self.destroy)


        sites = ['avito', 'youla', 'ebay', 'ufc_fights']
        for i, site in enumerate(sites):
            self_Radiobutton = tk.Radiobutton(self, text=site,
                                           command=self.onPressRadioBut,
                                           variable=self.site_radio_but,
                                           value=site)
            self_Radiobutton.grid(row=i, column=4)

        self.site_radio_but.set(sites[1])

        init_params_dict = Window.get_site_params(self.site_radio_but.get())
        self.setSiteParams(init_params_dict)


    def create_layout(self):
        padWE = dict(sticky=(tk.W, tk.E), padx="0.5m", pady="0.5m")

        self.start_url_Entry.grid(row=0,columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.page_param_Entry.grid(row=0, column=3, **padWE)


        self.container_el_Entry.grid(row=1, column=0, **padWE)
        self.el_Entry.grid(row=1, column=1, **padWE)
        self.pages_load_stop_Entry.grid(row=1,column=3,**padWE)

        self.container_pages_Entry.grid(row=2, column=0, **padWE)
        self.pages_Entry.grid(row=2, column=1, **padWE)

        self.save_Button.grid(row=3, column=0, **padWE)
        self.cncl_Button.grid(row=3, column=1, **padWE)

        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=2)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=1)
        #self.minsize(100, 20)


if __name__ == '__main__':
   # application = tk.Tk()
   # window=Window(application)

   # application.mainloop()

    a  = Window.get_site_params('avito')
    b = Window.get_site_params('ebay')

    #debug(Window.get_site_params,'avito')

   #Window.temp()
   #res = debug(Window.temp)