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
import structures.dfanalyser

def set_trace():
	from IPython.core.debugger import Pdb
	import sys
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args,**kwargs):
    from IPython.core.debugger import  Pdb
    pdb=Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args,**kwargs)

def patternSimTest(stb_name, pattern, list_func=[]):
        regex = re.compile(pattern, flags=re.IGNORECASE)
        ar = df[df[stb_name].str.contains(regex) == False]
        # self.df[stb_name] = self.df[stb_name].str.findall(regex) - изменить чуть паттерн, чтобы у нас чисто \d+
        return ar

def valuesSetTest(df,stb_name):
        ar = df[stb_name].value_counts()
        return ar

# def show_discharges(df, list_stbs, quantile_num=0.99):
#         df = df[list_stbs]
#         avito.avito_flat_parser.AvitoFlatParser.cleanPrepare(df, list_stbs)
#         for stb in list_stbs:
#             # df_disch_index = df[df[stb] > df[stb].quantile(quantile_num)].index
#             df_disch1 = df[df[stb] < df[stb].quantile(1 - quantile_num)]
#             df_disch2 = df[df[stb] > df[stb].quantile(quantile_num)]
#
#             df_disch = pd.concat([df_disch1, df_disch2], sort=False, ignore_index=True)
#             yield df_disch

def draw_distribution(df, list_stbs=[], cleaned=False,num_classes=20):
        if cleaned==False:
            avito.avito_flat_parser.AvitoFlatParser.cleanPrepare(df, list_stbs)
        if list_stbs:
            df = df[list_stbs]

        for i, stb in enumerate(df.columns.values):
            cutted = pd.cut(df[stb], num_classes, precision=2)
            cutted_stat = pd.value_counts(cutted)
            cutted_stat = cutted_stat.reset_index()
            list_rights = [item.right for item in cutted_stat['index']]
            ar_rights = np.array(list_rights)
            indexer = ar_rights.argsort()

            cutted_stat_sorted = cutted_stat.iloc[indexer]
            fig, ax = plt.subplots(1, 1)

            xs = [i * 10 for i, _ in enumerate(ar_rights[indexer])]
            ax.set_xticks([i * 10 for i, _ in enumerate(ar_rights[indexer])])
            ax.set_xticklabels(ar_rights[indexer])
            ax.set_title(stb)
            ax.bar(xs, cutted_stat_sorted[stb], 8)

def show_categorical_dependencies(df,x_stb,y_stb, bins=[]):
    if not bins:
        ar=df[x_stb].value_counts()
        groups = ar.reset_index()['index']
    else: groups=pd.cut(df[x_stb], bins)

    grouped = df[y_stb].groupby(groups).mean()
    #len(groups[groups.values.isnull()])
    grouped = grouped.reset_index()
    if not bins: x = grouped['index'].map(lambda x: str(x))
    else: x = grouped[x_stb].map(lambda x: str(x))
    plt.bar(x, grouped[y_stb])
    plt.show()

if __name__=='__main__':

    from importlib import reload

    reload(structures.dfanalyser)

    df = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\flats.csv')
    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'build_age', 'dist_cent', 'house_type','cl_adr'])
    reg, r2, df3,filtered_dict = structures.dfanalyser.analyse(df,structures.dfanalyser.filter_discharges,regr_type='linear',quantile_num=0.99,analyse_list=['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor'])
    # reg, r2, df3,filtered_dict = structures.dfanalyser.analyse(df, structures.dfanalyser.filter_discharges, regr_type='elastic', quantile_num=0.99)
    list_stbs = ['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num', 'build_age', 'dist_cent']

    df = df[list_stbs]
    avito.avito_flat_parser.AvitoFlatParser.cleanPrepare(df, list_stbs)


    df3=df.copy()
    #show_categorical_dependencies(df3,'total_square','cost',[0,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,1000])
    #show_categorical_dependencies(df3, 'rooms_num', 'cost', [0,1,2,3,4,5,6,7,8,10])
    #show_categorical_dependencies(df2, 'house_type', 'cost')
    # ['total_square','house_type, 'rooms_num', 'build_age', 'dist_cent']

    bins =[0,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,1000]

    groups=pd.cut(df3['total_square'], bins)
    groups.name = 'groups'
    #groups=pd.merge(groups.to_frame(),df3['total_square'].to_frame(),left_index=True,right_index=True)

    #groups.rename(columns={'total_square_x': 'groups', 'total_square_y': 'total_square'})


    def three_std(group):
        return 3*group.std()

    stats = df3['cost'].groupby(groups).agg(['mean','std',three_std])

    stats_group = pd.merge(groups.to_frame(),stats,left_on='groups',right_index=True)
    stats_group = stats_group.sort_index()
    # def group_stats(group):
    #     return {'mean':group.mean(),'std':group.std()}
    # stats = df3['cost'].groupby(groups).apply(group_stats).unstack()

    df_stats=pd.merge(df3[['cost', 'total_square']], stats_group,left_index=True,right_index=True)

    cost_doubt=df_stats.loc[np.abs(df_stats['cost'] - df_stats['mean'])>df_stats['three_std']]

    cost_doubt_indexes = df_stats.loc[np.abs(df_stats['cost'] - df_stats['mean'])>df_stats['three_std']].index



    # debug(DfTester.filter_discharges_func, df, ['cost','total_square', 'total_floors', 'floor', 'rooms_num', 'build_age', 'dist_cent'], 0.95)





