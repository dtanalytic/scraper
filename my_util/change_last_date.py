import tkinter as tk
import tkinter.ttk as ttk
import pickle
import os

from datetime import datetime

def construct_last_date(root_win):
    Window(root_win)

class Window(tk.Toplevel):

    def __init__(self, master):
        super().__init__(master)
        self.date = tk.StringVar()

        date_Entry = tk.Entry(self, textvariable=self.date, width=20)
        date_Label = ttk.Label(self, text='дата')

        fill_Button = tk.Button(self, text='fill', command=lambda: self.fill_field())

        save_Button=tk.Button(self, text='save', command = lambda: self.saveParams())

        cncl_Button = tk.Button(self, text='cancel', command=self.destroy)


        padWE = dict(sticky=(tk.W, tk.E), padx="0.5m", pady="0.5m")
        date_Entry.grid(row=0, column=0, **padWE)
        date_Label.grid(row=0, column=1, **padWE)

        fill_Button.grid(row=1, column=1, **padWE)

        save_Button.grid(row=2, column=0, **padWE)
        cncl_Button.grid(row=2, column=1, **padWE)

        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=2)

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        if os.path.exists('last_date'):
            self.init_fields('last_date')



    def fill_field(self):
        last_date = datetime.today()
        last_date = last_date.replace(second=0, microsecond=0)

        self.date.set(last_date)


    def init_fields(self,filename):
        with open(filename,'rb') as f:
            last_date = pickle.load(f)

        self.date.set(last_date)


    def saveParams(self):
        last_date_str = self.date.get()
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d %H:%M:%S')
        with open('last_date', 'wb') as f:
            pickle.dump(last_date, f)

        self.destroy()