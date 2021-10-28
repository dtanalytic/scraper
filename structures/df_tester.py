
import re
import pandas as pd
import sys
sys.path.append(r'')
from sklearn.model_selection import learning_curve,validation_curve

from my_util.df_util import df_drop
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
#from avito.avito_flat_parser import AvitoFlatParser
from sklearn.linear_model import ElasticNetCV
# import avito.avito_flat_parser
from sklearn.model_selection import KFold
from itertools import combinations

from scipy.spatial import distance
import os

def patternSimTest(df,stb_name, pattern):
        regex = re.compile(pattern, flags=re.IGNORECASE)
        ar = df[df[stb_name].str.contains(regex) == False]
        # self.df[stb_name] = self.df[stb_name].str.findall(regex) - изменить чуть паттерн, чтобы у нас чисто \d+
        return ar

def valuesSetTest(df,stb_name):
        ar = df[stb_name].value_counts()
        return ar


def make_two_duples_dfs(df):
    from datetime import date
    df_seconds=df[df.duplicated(['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])]
    df_clear_seconds= df.drop_duplicates(['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])
    df_seconds_return=pd.concat([df_seconds,df_clear_seconds], sort=False)
    df_firsts = df_seconds_return[df_seconds_return.duplicated(['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'adr'])]

    df_firsts=df_firsts.sort_values(by='adr')
    df_seconds = df_seconds.sort_values(by='adr')
    # df_firsts.to_csv('flats_duples1_{}.csv'.format(date.today().isoformat()), index=False)
    # df_seconds.to_csv('flats_duples2_{}.csv'.format(date.today().isoformat()), index=False)
    return df_firsts,df_seconds

def draw_distribution(df, list_stbs=[], cleaned=False, num_classes=10):
        if cleaned == False:
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


def show_categorical_dependencies(df, y_stb, x_stb, agg_funcs_list= ['mean', 'min', 'max'], bins=[]):
    if not bins:
        groups = df[x_stb].values
    else:
        groups = pd.cut(df[x_stb], bins)
        groups = pd.Series([item.right for item in groups])

    grouped = df[y_stb].groupby(groups).agg(agg_funcs_list)
    grouped.plot(kind='bar', title='dependency '+y_stb +" from "+ x_stb )

def x_std_disch_intervals(df, y_stb, x_stb, mult=3, bins=[]):
        def mult_std(group, mult):
            return mult * group.std()

        if not bins:
            #groups = df[x_stb].values
            groups = pd.Series(df[x_stb].values)
        else:
            groups = pd.cut(df[x_stb], bins)

        groups.name = 'groups'

        stats = df[y_stb].groupby(groups).agg(['mean', 'std', ('{}_std'.format(mult), lambda x:mult_std(x, mult))])

        stats_group = pd.merge(groups.to_frame(), stats, left_on='groups', right_index=True)
        stats_group = stats_group.sort_index()


        df_stats = pd.merge(df[[y_stb, x_stb]], stats_group, left_index=True, right_index=True)
        doubt = df_stats.loc[np.abs(df_stats[y_stb] - df_stats['mean']) > df_stats['{}_std'.format(mult)]]
        doubt_indexes = df_stats.loc[np.abs(df_stats[y_stb] - df_stats['mean']) > df_stats['{}_std'.format(mult)]].index

        return doubt,set(doubt_indexes)


def get_discharges(df, list_stbs, quantile_num=0.95):
    filtered_indexes_dict = dict()

    for stb_name in list_stbs:
        bigger_ind = list(df[df[stb_name] > df[stb_name].quantile(quantile_num)].index)
        lower_ind = list(df[df[stb_name] < df[stb_name].quantile(1 - quantile_num)].index)
        bigger_ind.extend(lower_ind)
        filtered_indexes_dict[stb_name] = bigger_ind

    set_ind = set()

    for key, val in filtered_indexes_dict.items():
        set_ind = set_ind.union(set(val))

    return filtered_indexes_dict, set_ind


def n_sim_obj(df, neighbours_matr_sorted, dists_matr, cur_el_index, num_obj):

    neigh = df.iloc[neighbours_matr_sorted[cur_el_index, 1:num_obj + 1]]
    neigh['dist'] = dists_matr[[cur_el_index], neighbours_matr_sorted[cur_el_index, 1:num_obj + 1]]

    return neigh


def get_neigh_dists_matrs(df):
    # WITHOUT COST STB

    dists = distance.pdist(df, 'euclidean')
    dists = distance.squareform(dists)
    neighbours = dists.argsort(axis=1)

    return neighbours,dists

def draw_learning_curve(estim,x,y,sizes_l):
    train_sizes,train_scores, test_scores = learning_curve(estim, x, y, train_sizes=sizes_l,cv=5,n_jobs=-1)
    train_mean = np.mean(train_scores,axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis =1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes,train_mean, color='blue', marker='o', markersize=5, label = 'правильность при обучении')
    plt.fill_between(train_sizes, train_mean+train_std , train_mean - train_std,alpha = 0.2,color='blue')
    plt.plot(train_sizes,test_mean, color='green',marker='s', markersize=5, label='правильность при проверке')
    plt.fill_between(train_sizes, test_mean+test_std, test_mean - test_std, alpha=0.2, color = 'green')
    plt.grid()
    plt.xlabel('количество обучающих образцов')
    plt.ylabel('правильность')
    plt.legend()
    plt.ylim([0.5, 1])
    plt.show()

def draw_validation_curve(estim,x,y,param_name, param_range):

    train_scores, test_scores = validation_curve(estim, x,y, param_name=param_name,
                                                 param_range=param_range, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='правильность при обучении')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, marker='s', label='правильность при проверке', markersize=5, color='green')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.grid()
    plt.legend()
    plt.xlabel('параметр {}'.format(param_name))
    plt.ylabel('правильность')
    plt.ylim([0.5, 1])
    plt.show()


def form_indic_pattern_center_w_around_w_l(search_series,word_cent,around_w_l):
    ind_final = pd.Series([False]*len(search_series))
    ind_temp = pd.Series([False]*len(search_series))

    for word in around_w_l: 	
         # templ_l = re.compile(r'{}[а-я]*\s+{}[а-я]*'.format(word,word_cent))
         # templ_r = re.compile(r'{}[а-я]*\s+{}[а-я]*'.format(word_cent,word))
         templ_l = re.compile(r'{}[а-я.-]*\s*{}[а-я.-]*'.format(word,word_cent))
         templ_r = re.compile(r'{}[а-я.-]*\s*{}[а-я.-]*'.format(word_cent,word))
         ind_temp = np.any(pd.concat([search_series.str.contains(templ_l), search_series.str.contains(templ_r)],axis=1),axis=1)
         ind_final = np.any(pd.concat([ind_final, ind_temp],axis=1),axis=1)
         
    return ind_final.astype(int)
	
def vis_pattern_center_w(search_series,word_cent):
 	findings = df['desc'].str.findall(r'([А-Яа-я]*)\s+({}[а-я]*)\s+([А-Яа-я]*)\s+'.format(word_cent))
 	return findings[findings.map(lambda x: len(x)>0)]

def square_clean(x,pat):
        x = x.replace(',','.')
        # str = str.replace(' ','')
        x = pat.sub('',x)
        return x
    
if __name__=='__main__':

    from structures.df_analyser import DfFlatsPrepareSteps

    df = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats.csv')

    df= df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num','house_type', 'adr'])

    if os.path.exists(r'c:\work\dev\python\progs\scraper\flats_add.csv') and os.path.getsize(
            r'c:\work\dev\python\progs\scraper\flats_add.csv') > 15:
        df_add = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats_add.csv')

    flats_frame = pd.concat([df_add, df], sort=False, ignore_index=True)
    flats_frame.to_csv('flats.csv', index=False)
    
    

    # avito.avito_flat_parser.AvitoFlatParser.flag_continue_searching_dist=False
    # avito.avito_flat_parser.AvitoFlatParser.calc_distance_center(df, 0, 10, 'Владикавказ', (43.024531, 44.682651))
    # first, last = 0,17470
    # df.loc[range(first, last), 'cl_adr']=df.loc[range(first, last), 'adr'].map(avito.avito_flat_parser.AvitoFlatParser.streetNumber,str)
    # avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(df, first, last,dict_house_age)
    # df.to_csv('flats.csv', index=False)
    
    
    # df_flat_age = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats_age.csv', sep=';')
    # pat = re.compile(r'[^0-9\.]')
    # df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)'].astype(str) 
    # df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)'].map(lambda x: square_clean(x, pat))
  
    # dict_house_age = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age,'год постройки')
    # dict_house_sq = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age,'общая площадь(кв.м.)')

    
    
    # df['house_sq'] = df['cl_adr'].map(dict_house_sq)
    
    
    # df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].map(lambda x: x[0])
    
    # # df.loc[df['house_sq'].isnull(), 'house_sq'] = df.loc[df['house_sq'].isnull(), 'cl_adr'].map( \
    # #         lambda x: avito.avito_flat_parser.AvitoFlatParser.levenDistForNull(x, dict_house_sq))
    
    # ind_empty = df.loc[df['house_sq']==''].index
    # df = df.drop(ind_empty)
    # df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].astype(np.float32) 
    
    
    

    # new_build = df['desc'].str.contains('новостройк').astype(int)
    # new_build.name='new_build'
    # df = pd.concat([df,new_build],axis=1)
    
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep'],\
    #                                  drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                )
        
        
    # a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                      'house_type', 'is_lst_floor'], \
    #                                 drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                                      'house_type']
    #                                 )
        
    

    
    # s = df['desc'][elite_rep]
    # vis = vis_pattern_center_w(df['desc'],'элитн')
    
    

    # a.analyse_1steps(quantile_num= 0.99,std_num = 2)
    # a.normalize()
    # a.join_parts()


    # df=a.df
    # corr = a.df.corr()

    # col_func_dict={'total_square':np.square}
    # df= add_cols_func_list(df, col_func_dict)
    #


    # l=np.array([1,2,3,4,5,6,7,8])
    # np.square(l)

    # show_categorical_dependencies(a.df, 'cost', 'total_square', agg_funcs_list=['mean'], bins=30)
    #
    # show_categorical_dependencies(a.df[a.df['floor']<=16], 'cost', 'floor', agg_funcs_list=['mean'])
    # show_categorical_dependencies(a.df[a.df['total_floors']<=16], 'cost', 'total_floors', agg_funcs_list=['mean'])
    # show_categorical_dependencies(a.df[a.df['rooms_num']!=8, 'cost', 'rooms_num', agg_funcs_list=['mean'])

    # show_categorical_dependencies(a.df, 'cost', 'dist_cent', agg_funcs_list=['mean'], bins=3)
    # show_categorical_dependencies(a.df, 'cost', 'build_age', agg_funcs_list=['mean'], bins=100)


