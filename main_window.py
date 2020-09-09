from my_util.df_util import df_drop
import tkinter.ttk as ttk
import tkinter as tk
import ctypes
import os
import pickle
import preferences_window
import threading
import avito.avito_flat_parser
from pandas import DataFrame
import pandas as pd
import queue
import locale
from datetime import date, datetime,timedelta
import shutil
from shop_product_parser import StopException
from my_util.change_page_params import construct_page_params
from my_util.change_last_date import construct_last_date
from tkinter import messagebox

import logging

fmt='%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(level='INFO', filename='work_meta.log',format=fmt)
logger = logging.getLogger('scraper_log')


def writeDfThread(page_parser,messages_queue, product_last_datetime):

   try:

        page_parser.startListProductsParser(messages_queue, product_last_datetime)
        # в принципе сюда попадем только если функция сверху завершится без исключений, например StopException. но есть еще случай - если достигнем количество скачанных страниц (поэтому там ставим pause_flag)
        if not page_parser.pause_flag:
            flats_frame = DataFrame(page_parser.products_list)

            #flats_frame = ''
            df_flat_age = pd.read_csv(page_parser.flats_age_filename, sep=';')
            dict_house_age = avito.avito_flat_parser.AvitoFlatParser.get_house_age(df_flat_age)

            logger.info('complete downloading flats_frame size - {}'.format(len(flats_frame)))


            if os.path.exists('flats_add.csv') and os.path.getsize('flats_add.csv')>10:
                old_flats_frame = pd.read_csv('flats_add.csv')
                logger.info('flats_add exists it has size - {}'.format(len(old_flats_frame)))
                if len(flats_frame) != 0:
                    flats_frame = pd.concat([flats_frame, old_flats_frame], sort=False, ignore_index=True)
                else:
                    flats_frame = old_flats_frame
                flats_frame.to_csv('ark_add/flats_add {}.csv'.format(date.today().isoformat()), index=False)
                logger.info('unified flats_frame and flats_add size - {}'.format(len(flats_frame)))
                os.remove('flats_add.csv')
                logger.info('flats_add removed')


            old_size = 0
            new_size = 0
            if os.path.exists('flats.csv'):
                shutil.copy('flats.csv', 'ark/flats {}.csv'.format(date.today().isoformat()))
                old_flats_frame = pd.read_csv('flats.csv')
                old_size = len(old_flats_frame)
                logger.info('flats.csv exists has size - {}'.format(len(old_flats_frame)))
                if len(flats_frame) != 0:
                    flats_frame = pd.concat([flats_frame, old_flats_frame], sort=False, ignore_index=True)
                else:
                    flats_frame = old_flats_frame
                logger.info('flats.csv with new records has size - {}'.format(len(flats_frame)))


            flats_frame = df_drop(flats_frame,list_duples=['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])
            logger.info('flats.csv after dropping duples has size - {} and duples_num - {}'.format(len(flats_frame),flats_frame[flats_frame.duplicated(['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])].shape[0]))
            new_size = len(flats_frame)

            #flats_frame.to_csv('flats_bef_ya.csv', index=False)
            flats_frame.to_csv('flats.csv', index=False)
            logger.info('new_size - {} versus old_size - {}'.format(new_size, old_size))

            if len(flats_frame) != 0:

                # есть что-то новое
                if new_size > old_size:

                    logger.info('start collecting dists')
                    #порядок именно такой, чтобы cl_adr считался в calc_distance_center
                    # avito.avito_flat_parser.AvitoFlatParser.calc_distance_center(flats_frame, 0, new_size - old_size,\
                    #                                                              page_parser.city_name, \
                    #                                                              page_parser.city_cent)
                    logger.info('start collecting build_ages')
                    # avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(flats_frame, 0, new_size - old_size,
                    #                                                        dict_house_age)


                    # запуск функций от 0 до new_size-old_size
                    flats_frame.to_csv('flats.csv', index=False)

                    logger.info('flats.csv after calcs has size - {} and duples_num - {} '.format(len(flats_frame),flats_frame[flats_frame.duplicated(['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])].shape[0]))


            # задали last_date в самом начале скачки
            with open('last_date', 'wb') as file:
                last_date = datetime.today()
                last_date_s = last_date.strftime('%Y-%m-%d %H:%M:%S')
                last_date = datetime.strptime(last_date_s, '%Y-%m-%d %H:%M:%S')
                    # last_date =last_date.replace(hour=0,minute=0, second=0, microsecond=0)
                pickle.dump(last_date, file)


            # чтобы репитер поменял надпись на кнопке на старт
            page_parser.pause_flag=True
            messages_queue.put('обработка завершена, можно выйти из программы')

   except (StopException,Exception) as e:

       if not StopException:

           messages_queue.put('Исключение {}'.format(e))

       flats_frame = DataFrame(page_parser.products_list)

       if product_last_datetime:
           flats_file_name = 'flats_add.csv'
       else:
           flats_file_name = 'flats.csv'

       if os.path.exists(flats_file_name) and os.path.getsize(flats_file_name)>10:
           logger.info('Exception - {} not saved records with {}'.format(len(flats_frame),flats_file_name))
           old_flats_frame = pd.read_csv(flats_file_name)
           # порядок конкатенации меняем потому что есди у нас с даты идет скачивание, то добавляются новые, а уже к ним
           # старье, иначе качаем старые и добавляются к предыдущим снизу
           flats_frame = pd.concat([old_flats_frame, flats_frame], sort=False, ignore_index=True)
           logger.info('with saved {} and new recs unifyed length is - {}'.format(len(old_flats_frame),len(flats_frame)))

       logger.info('Запись файла {} внутри исключения'.format(flats_file_name))
       flats_frame.to_csv(flats_file_name, index=False)

       with open('params_page_parser', 'wb') as f:
           tuple_params = page_parser.getClassParams()
           pickle.dump(tuple_params, f)

       messages_queue.put('остановка')
       messagebox.showinfo('остановка')
       # ctypes.windll.user32.MessageBoxW(0, "Stop Exception", "Warning!", 32)
        #messages_queue.task_done()


def outer_f():

        pause_flag = [False]
        thread = [None]
        def inner_f(button, page_parser_inner,messages_queue,product_last_date_var,from_date_flag):
            #import sys
            #sys.setrecursionlimit(100000)

            #если выполнение программы остановлено по достижении конца или заданного количества страниц, то
            #поток сам остановится, а надпись на старт изменится, поэтому флаг надо поменять как быдто кнопка пауза
            # была нажата
            if page_parser_inner.pause_flag:
                pause_flag[0] = not pause_flag[0]

            if pause_flag[0]:
                page_parser_inner.pause_flag = True
                thread[0].join()

                button.config(text='start')
                page_parser_inner.pause_flag = False
            else:
                if os.path.exists('params_page_parser'):

                    with open('params_page_parser', 'rb') as f:
                        tuple_params = pickle.load(f)
                        page_parser_inner.setClassParams(tuple_params)

                    os.remove('params_page_parser')

                if from_date_flag and not product_last_date_var.get()=='':
                    product_last_datetime = datetime.strptime(product_last_date_var.get(), '%Y-%m-%d %H:%M:%S')
                else: product_last_datetime=''

                thread[0] = threading.Thread(target=writeDfThread,args=(page_parser_inner,messages_queue, product_last_datetime))  # run вызовет target
                thread[0].daemon = True
                thread[0].start()

                button.config(text='pause')
            pause_flag[0] = not pause_flag[0]

        return inner_f

def startPreferences(root_win):
        preferences_window.Window(root_win)
        root_win.focus()


class Window(ttk.Frame):

    def __init__(self,master):
        super().__init__(master)

        start_pause = outer_f()

        self.messages_queue = queue.Queue()


        #self.page_parser = avito_flat_parser.AvitoFlatParser(**preferences_window.Window.getSiteParams('youla'))

        menubar = tk.Menu(master)
        master.config(menu=menubar)

        preferences_Menu = tk.Menu()
        preferences_Menu.add_command(label="Preferences", underline=0,
                                     command=lambda: startPreferences(self))



        preferences_Menu.add_separator()
        preferences_Menu.add_command(label="Change_page_params", underline=0,
                                     command=lambda:construct_page_params(self))

        preferences_Menu.add_separator()
        preferences_Menu.add_command(label="Change_last_date", underline=0,
                                     command=lambda:construct_last_date(self))

        preferences_Menu.add_separator()
        preferences_Menu.add_command(label="Quit", underline=0, command=master.destroy)
        menubar.add_cascade(label="File", underline=0, menu=preferences_Menu)



        start_pause_Button = tk.Button(master, text='start', command=lambda: start_pause(start_pause_Button, self.page_parser,self.messages_queue,self.product_last_date_var,self.from_date_flag_var))
        start_pause_Button.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))

        quit_pause_Button = tk.Button(master, text='Quit', command=master.destroy)
        quit_pause_Button.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))


        self.status = tk.StringVar()
        status_Label = ttk.Label(master, textvariable=self.status, text='статус')
        status_Label.grid(row=3, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))

        self.product_last_date_var = tk.StringVar()
        product_last_date_Entry = tk.Entry(master, width=30, textvariable=self.product_last_date_var)
        product_last_date_Entry.grid(row=1, columnspan=2, sticky=(tk.N, tk.S, tk.E, tk.W))

        #product_last_date_Entry.config(state='disabled')

        self.from_date_flag_var = tk.IntVar()
        from_date_CheckBox = tk.Checkbutton(master, variable = self.from_date_flag_var, text='с',command=lambda:product_last_date_Entry.config(state='normal') if self.from_date_flag_var.get()==1 else product_last_date_Entry.config(state='disabled'))
        from_date_CheckBox.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.from_date_flag_var.set(value=1)

        fill_Button = tk.Button(master, text='fill', command=lambda: self.fill_field())
        fill_Button.grid(row=2, column=2, sticky=(tk.N, tk.S, tk.E, tk.W))

        master.rowconfigure(0, weight=1)
        master.rowconfigure(1, weight=1)
        master.rowconfigure(2, weight=1)
        master.rowconfigure(3, weight=1)

        master.columnconfigure(0, weight=1)
        master.columnconfigure(1, weight=1)
        master.columnconfigure(2, weight=1)
        master.columnconfigure(3, weight=1)

        self.page_parser = preferences_window.Window.getSiteClass()
        #self.page_parser=avito_flat_parser.AvitoFlatParser(**preferences_window.Window.getSiteParams('avito'))

        self.messages_queue.put('начальный статус')
        self.repeater(start_pause_Button,status_Label)


        if os.path.exists('last_date'):
            with open('last_date', 'rb') as f:
                self.product_last_date_var.set(pickle.load(f))


    def fill_field(self):
        last_date = datetime.today()
        product_last_datetime_s = last_date.strftime('%Y-%m-%d %H:%M')
        self.product_last_date_var.set(product_last_datetime_s)

    def repeater(self, button,status_Label):
        if self.page_parser.pause_flag:
            button.config(text='start')

        try:
            msg = self.messages_queue.get_nowait()
            self.status.set(msg)
        except:
            pass#print('empty')
           # self.messages_queue.task_done()

        status_Label.after(1000, self.repeater, button,status_Label)