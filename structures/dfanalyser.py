import re
import pandas as pd
from my_util.df_util import df_drop

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#from avito.avito_flat_parser import AvitoFlatParser
from sklearn.linear_model import ElasticNetCV
import avito.avito_flat_parser
from sklearn.model_selection import KFold
from itertools import combinations


def set_trace():
	from IPython.core.debugger import Pdb
	import sys
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args,**kwargs):
    from IPython.core.debugger import  Pdb
    pdb=Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args,**kwargs)

def remember_stbs(df):
    flat_type = pd.get_dummies(df['house_type'])
    is_lst_floor = np.where(df['total_floors'] == df['floor'], 1, 0)
    is_lst_floor=pd.DataFrame(is_lst_floor)
    return is_lst_floor, flat_type

def draw_dependency(predicted, real):
    plt.scatter(predicted, real)
    plt.xlabel('предсказанная цена')
    plt.ylabel('фактическая цена')
    plt.plot([real.min(), real.max()], [[real.min()], [real.max()]])
    plt.show()


def mean_cross_valid_error(y_stb, table_x, reg_method, num_classes=10):
    kf = KFold(n_splits=num_classes, shuffle=True)
    # kf.get_n_splits(df4)
    errors_r2 = []

    for training, testing in kf.split(table_x):
        x = table_x.iloc[training]
        y = y_stb[training]
        reg_method.fit(x, y)
        r2 = r2_score(y_stb[testing], reg_method.predict(table_x.iloc[testing]))
        errors_r2.append(r2)

    return np.mean(errors_r2)


def findBestParamsLrModel(y_stb, table_x, stb_out_list, reg_method):
    reg_method.fit(table_x, y_stb)
    reg_method_best = reg_method
    r2_best = mean_cross_valid_error(y_stb, table_x, reg_method)
    list_stb_best = stb_out_list.copy()

    # for i in range(1, len(stb_out_list)):
    for i in range(1, 2):
        for j in combinations(stb_out_list, i):
            if i == 1:
                j = j[0]
            else:
                j = list(j)
            df_out = table_x.drop(j, axis=1)
            reg_method.fit(df_out, y_stb)
            r2 = mean_cross_valid_error(y_stb, df_out, reg_method)
            # print('stb list = {}'.format([l for l in stb_out_list if l not in j]), end=" ")

            # print('mean error = {}'.format(r2))
            if r2 > r2_best:
                # print('смена лучшей')
                r2_best = r2
                reg_method_best = reg_method
                list_stb_best = [l for l in stb_out_list if l not in j]

    return reg_method_best, r2_best, list_stb_best


def analyse(df, \
            filter_discharges_func,  \
            analyse_list=['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','build_age','dist_cent','house_type','is_lst_floor'],\
            quantile_num=0.99,\
            regr_type='elastic', \
            ):
    list_stbs = analyse_list.copy()
    if 'build_age' in analyse_list:
        #df = df.loc[df['build_age'].notnull()]
        df=df_drop(df, nn=True, stb='build_age')

    if 'dist_cent' in analyse_list:
        #df = df.loc[df['dist_cent'].notnull()]
        df=df_drop(df, nn=True, stb='dist_cent')

    if 'is_lst_floor' in analyse_list or 'house_type' in analyse_list:
        is_lst_floor, flat_type = remember_stbs(df)

        if 'is_lst_floor' in list_stbs: list_stbs.remove('is_lst_floor')
        if 'house_type' in list_stbs: list_stbs.remove('house_type')

    df = df[list_stbs]

    avito.avito_flat_parser.AvitoFlatParser.cleanPrepare(df, list_stbs)

    if 'live_square' in analyse_list:
        bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
        cat = pd.cut(df['total_square'], bins)
        live_sq = df['live_square'].groupby(cat).transform(np.mean)
        df['live_square'] = np.where(df['live_square'].isnull(), live_sq, df['live_square'])

    # df = filter_discharges_func(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num', 'build_age', 'dist_cent'], quantile_num)

    # df = df[df['cost'] < 6000000]
    # df = df[df['cost'] > 1.500000]
    # df = df.reset_index()
    # df = df.drop('index', axis=1)


    list_for_filtering = analyse_list.copy()
    for item in ['live_square', 'house_type', 'is_lst_floor', 'total_floors', 'floor', 'rooms_num', 'build_age',
                 'dist_cent']:
        if item in list_for_filtering: list_for_filtering.remove(item)
    df, filtered_indexes_dict, filtered_indexes_set = filter_discharges_func(df, list_for_filtering, quantile_num)
    df_to_return=df.copy()
    df = (df - df.mean()).div(df.std())

    # df2=df.iloc[range(0,3)]
    # df2 = df2.drop('house_type', axis=1)
    #
    # df2 = (df2 - df2.mean()).div(df2.std())
    # mean1=df2.sum().div(df2.count())
    # df4=np.square(df3)
    # #std1=np.sqrt(df4.sum().div(df4.count()-1)+0.1)
    #set_trace()
    if 'house_type' in analyse_list:
        #set_trace()
        flat_type = df_drop(flat_type, list_indexes=filtered_indexes_set)
        df = df.join(flat_type)
        df_to_return= df_to_return.join(flat_type)
    if 'is_lst_floor' in analyse_list:

        is_lst_floor = df_drop(is_lst_floor, list_indexes=filtered_indexes_set)

        df['is_lst_floor'] = is_lst_floor
        df_to_return['is_lst_floor'] = is_lst_floor

    # перед перекрестной проверкой надо делать df3.reset_index() и удаление стб индекса чтобы не появились пустые записи, видимо выбирает порядковые номера
    # а после фильтрации некоторые порялковые номера из by индекса становятся пустыми

    df = df_drop(df, na=True)

    # df3 = df
    cost = df['cost']
    x = df.drop(['cost'], axis=1)
    y = cost

    if regr_type == 'elastic':
        reg = ElasticNetCV()
    if regr_type == 'linear':
        reg = LinearRegression()

    reg.fit(x, y)
    predicted = reg.predict(x)
    # r2 = r2_score(y,predicted)

    r2 = mean_cross_valid_error(y, x, reg)

    # analyse_list.remove('cost')
    # if 'house_type' in analyse_list:
    #     analyse_list.remove('house_type')
    # reg_method, r2, list_stb_best = DfTester.findBestParamsLrModel(y, x, analyse_list,reg)

    draw_dependency(predicted, y)

    # mse = mean_squared_error(y, reg.predict(x))
    # nprmse = np.sqrt(mse)

    return reg, r2, df_to_return,filtered_indexes_dict  # , list_stb_best



def filter_discharges(df, list_stb_name, quantile_num):
        filtered_indexes_dict = dict()

        for stb_name in list_stb_name:
            bigger_ind=list();lower_ind=list()
            bigger_ind = list(df[df[stb_name] > df[stb_name].quantile(quantile_num)].index)
            lower_ind = list(df[df[stb_name] < df[stb_name].quantile(1 - quantile_num)].index)
            bigger_ind.extend(lower_ind)
            filtered_indexes_dict[stb_name] = bigger_ind

        set_ind = set()
        set_ind_inter = set(filtered_indexes_dict['cost'])
        for key, val in filtered_indexes_dict.items():

            set_ind = set_ind.union(set(val))
            set_ind_inter = set_ind_inter.intersection(set(val))

        #df=df.drop(set_ind)
        df = df_drop(df, list_indexes=set_ind)
        #return df,filtered_indexes_dict,set_ind, set_ind_inter
        return df,filtered_indexes_dict,set_ind


class DfPrepareSteps():

    @classmethod
    def from_df(cls,df,analyser_cols):
        obj=cls()
        obj.df=df
        obj.df_parts=[]
        obj.analyzer_cols=analyser_cols
        return obj

    def df_drop_nulls(self,df,drop_cols_list):

        set_ind = set()

        for stb_name in drop_cols_list:

            set_ind_new=set(df[df[stb_name].isnull()].index)
            set_ind.update(set_ind_new)
            #set_trace()
        self.df=df_drop(df, list(set_ind))
        if self.df_parts:
            for df_p in self.df_parts:
                df_p=df_drop(df_p, list(set_ind))

        return self.df


    def get_drop_list(self):
        return self.analyser_cols

class DfFlatsPrepareSteps(DfPrepareSteps):

    def get_drop_list(self):

        #return ['build_age','dist_cent']
        return self.ana

if __name__=='__main__':

   # from importlib import reload

    #reload(my_util.df_util)

    df = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\flats.csv')
    #df= df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'build_age', 'dist_cent', 'house_type','cl_adr'])


    a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor'])

    #set_trace()
    df3=a.df_drop_nulls(a.df,a.get_drop_list())