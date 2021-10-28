import pandas as pd
from my_util.df_util import df_drop
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from structures.df_analyser import DfFlatsPrepareSteps
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score
from structures.df_tester import draw_learning_curve,draw_validation_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from structures.df_tester import get_discharges,x_std_disch_intervals
from sklearn.decomposition import PCA
import avito.avito_flat_parser
from geopy import distance
import re
import locale
from structures.df_tester import form_indic_pattern_center_w_around_w_l,square_clean,vis_pattern_center_w
from contextlib import contextmanager
from datetime import datetime
import os

@contextmanager
def set_rus_locale():
    locale.setlocale(locale.LC_ALL, '')
    try:        
        yield
    finally:    
        # locale.setlocale(locale.LC_ALL, 'en')
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')
        
def date_time_tr(s,month_dict):
    if os.name == 'posix':
        month_dict = {v:k for k,v in month_dict.items()}
    f_gr = s.split()    
    mon_new = month_dict.get(f_gr[1].lower(),f_gr[1].lower())
    return '{} {} {} {}'.format(f_gr[0],mon_new,f_gr[2],f_gr[3])



def get_regr_grid_estim(reg, grid_params, x_train, y_train):
    grid_search = GridSearchCV(reg, grid_params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    reg = grid_search.best_estimator_
    return reg


if __name__=='__main__':
    
    df = pd.read_csv(r'../data/flats.csv')
    
    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 
                                  'rooms_num', 'house_type', 'adr'])
    
    
    
    # дату и время добавляем и преобразум к типу даты
    month_dict = {'января':'январь', 'февраля':'февраль','марта':'март','апреля':'апрель',
                  'мая':'май','июня':'июнь','июля':'июль','августа':'август',
                  'сентября':'сентябрь','октября':'октябрь','ноября':'ноябрь','декабря':'декабрь'}

    df.loc[df.date_time.notnull(),'date_time'] = df.loc[df.date_time.notnull(),'date_time']\
                                            .map(lambda s: date_time_tr(s,month_dict))
    
    with set_rus_locale():
        df.loc[df.date_time.notnull(),'date_time'] = df.loc[df.date_time.notnull(),'date_time']\
                            .map(lambda s: datetime.strptime(s, '%d %B %H:%M %Y'))
        df['date_time'] = df['date_time'].astype(np.datetime64)
    
    
    # заполняем расстояние от центра города
    first, last = 0,4708
    df.loc[range(first, last), 'dist_cent'] = df.loc[range(first, last), ['lat','lon']].apply(lambda x: distance.distance((43.024531, 44.682651), (x[0], x[1])).km,axis=1)
    df.loc[df['lat'].isnull()].index
    
    
    # заполняем общую площадь и год постройки дома    
    df_flat_age = pd.read_csv(r'../data/flats_age.csv', sep=';')
    pat = re.compile(r'[^0-9\.]')
    df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)']\
                                        .astype(str) 
    df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)']\
                                        .map(lambda x: square_clean(x, pat))
  
    dict_house_age = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age.copy(),'год постройки')
    dict_house_sq = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age.copy(),'общая площадь(кв.м.)')
    
    df.loc[range(first, last), 'cl_adr']=df.loc[range(first, last), 'adr']\
                    .map(avito.avito_flat_parser.AvitoFlatParser.streetNumber)
    
    avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(df, first, last,dict_house_age)
    
        
    df['house_sq'] = df['cl_adr'].map(dict_house_sq)
    df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].map(lambda x: x[0])
    df = df.drop(df.loc[df['house_sq']==''].index)
    df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].astype(np.float32) 
 

    # формируем индикаторы наличия ремонта    
    around_w_l=['дорог', 'капитальн', 'шикарн', 'евро', 'кап', 'новы', 'идеальн','полн','современн']
    elite_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    elite_rep.name='elite_rep'
    df = pd.concat([df,elite_rep],axis=1)
    
    x = np.arange(1,22) 
    y = np.sum(2*x+4)
    
    
    a = {}
    # around_w_l=['хорош', 'косметическ']
    # good_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    # good_rep.name='good_rep'
    # df = pd.concat([df,good_rep],axis=1)
    
    # # df.to_csv('flats.csv', index=False)
    # # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep','good_rep','date_time','house_sq'],\
    # #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent','house_sq']
    # #                                 )

    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','elite_rep','good_rep','house_sq'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','elite_rep','good_rep','house_sq']
    #                                 )
    
    # # a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    # #                                       'house_type', 'is_lst_floor'], \
    # #                                 drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    # #                                                       'house_type']
    # #                                 )
        
    # a.df_drop_nulls()
    # a.split_df_preserve_cols()
    # a.ready_analyse()
    # a.transform_cols()
    
    
    # filt_dict, dis_ind = get_discharges(a.df, a.list_for_filtering, quantile_num=0.99)
    # _, doubt_indexes = a.std_specific_disch(std_num=2)
    # dis_ind = dis_ind.union(doubt_indexes)
    # a.df_with_parts_drop_ind(list(dis_ind))
    
    


    # bins = [1500000, 3000000, 4500000, 7000000, 100000000]
    # cat = pd.cut(a.df['cost'], bins)
    
    # a.normalize()
    # a.join_parts()

    # categ_fr = pd.get_dummies(cat)
    # a.df = pd.merge(a.df,categ_fr,left_index=True,right_index=True)

    # x = a.df.drop(['cost'], axis=1)
    # y = a.df['cost'].copy()
    
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # # pca = PCA(0.95)
    # # x_train=pca.fit_transform(x_train)
    # # x_test = pca.transform(x_test)


    # # reg = ElasticNet()
    # # reg = RandomForestRegressor(n_estimators=5, min_samples_leaf=2, max_features=15,max_leaf_nodes=400,max_depth=18)
    # # reg = RandomForestRegressor(n_estimators=100, max_features=12,max_depth=15)
    # reg = RandomForestRegressor(n_estimators=300,max_depth= 17)
    
    
    # # tr_grid_params=[{'n_estimators':[100,120,150], 'max_features':[12,15,18], 
    # #               'max_depth':[15,18,22]}]
    # # tr_grid_params=[{'n_estimators':[100,300,500],'max_depth':[15, 17]}]
    # # el_grid_params=[{'alpha':[0.00001,0.0001,0.001]}]
    # # reg = get_regr_grid_estim(reg,tr_grid_params,x_train,y_train)
    
    # reg.fit(x_train,y_train)
    
    
    # # from sklearn.feature_selection import RFECV
    # # rfecv = RFECV(estimator=reg, step=1, scoring="r2",cv=3,n_jobs=-1)
    # # rfecv.fit(x, y)
    # # rfecv.support_
    # # x_new = rfecv.transform(x)
    # # np.array(x.columns)[~rfecv.support_]
    
    # # draw_learning_curve(reg,x,y, np.linspace(0.1,1.0,10))
    # # draw_validation_curve(reg, x,y, param_name='n_estimators',param_range=[3,10, 40])
    # # draw_validation_curve(reg, x,y, param_name='alpha',param_range=np.linspace(1e-15,1e-10,19))
     
     
    # y_pr = reg.predict(x_test)
    # rmse = np.sqrt(mean_squared_error(y_pr, y_test))
    # dif = rmse*a.df_std['cost']
    # print(r2_score(y_pr, y_test))
    # # print(np.mean(cross_val_score(reg,x_test,y_test, cv=3, scoring='r2')))
    

