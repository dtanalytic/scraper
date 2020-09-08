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

@contextmanager
def set_rus_locale():
    locale.setlocale(locale.LC_ALL, '')
    try:        
        yield
    finally:    
        # locale.setlocale(locale.LC_ALL, 'en')
        locale.setlocale(locale.LC_ALL, 'en_US.utf8')
        
def date_time_tr(s,month_dict):
    f_gr = s.split()    
    mon_new = month_dict.get(f_gr[1].lower(),f_gr[1].lower())
    return '{} {} {} {}'.format(f_gr[0],mon_new,f_gr[2],f_gr[3])



def get_regr_grid_estim(reg, grid_params, x_train, y_train):
    grid_search = GridSearchCV(reg, grid_params, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    # feature_importances = grid_search.best_estimator_.feature_importances_
    # print('feature_importances \n{}'.format(feature_importances))
    # cv_res = grid_search.cv_results_
    # for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    #     print(np.sqrt(-mean_score), params)
    reg = grid_search.best_estimator_
    return reg


    
if __name__=='__main__':
    
    df = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats.csv')

    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'adr'])
    # df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'desc'])
    
    # дату и время добавляем и преобразум к типу даты
    month_dict = {'января':'январь', 'февраля':'февраль','марта':'март','апреля':'апрель',
                  'мая':'май','июня':'июнь','июля':'июль','августа':'август',
                  'сентября':'сентябрь','октября':'октябрь','ноября':'ноябрь','декабря':'декабрь'}

    df.loc[df.date_time.notnull(),'date_time'] = df.loc[df.date_time.notnull(),'date_time'].map(lambda s: date_time_tr(s,month_dict))
    
    with set_rus_locale():
        df.loc[df.date_time.notnull(),'date_time'] = df.loc[df.date_time.notnull(),'date_time'].map(lambda str: datetime.strptime(str, '%d %B %H:%M %Y'))
        df['date_time'] = df['date_time'].astype(np.datetime64)
    
    # заполняем расстояние от центра города
    first, last = 0,4069
    df.loc[range(first, last), 'dist_cent'] = df.loc[range(first, last), ['lat','lon']].apply(lambda x: distance.distance((43.024531, 44.682651), (x[0], x[1])).km,axis=1)
    
    # заполняем адрес, год сдачи дома и общую площадь    
    df_flat_age = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats_age.csv', sep=';')
    pat = re.compile(r'[^0-9\.]')
    df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)'].astype(str) 
    df_flat_age['общая площадь(кв.м.)']= df_flat_age['общая площадь(кв.м.)'].map(lambda x: square_clean(x, pat))
  
    dict_house_age = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age.copy(),'год постройки')
    dict_house_sq = avito.avito_flat_parser.AvitoFlatParser.get_house_param(df_flat_age.copy(),'общая площадь(кв.м.)')
    
    df.loc[range(first, last), 'cl_adr']=df.loc[range(first, last), 'adr'].map(avito.avito_flat_parser.AvitoFlatParser.streetNumber,str)
    
    avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(df, first, last,dict_house_age)
    
        
    df['house_sq'] = df['cl_adr'].map(dict_house_sq)
    df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].map(lambda x: x[0])
    df = df.drop(df.loc[df['house_sq']==''].index)
    df.loc[df['house_sq'].notnull(), 'house_sq'] = df.loc[df['house_sq'].notnull(), 'house_sq'].astype(np.float32) 
    
    around_w_l=['дорог', 'капитальн', 'шикарн', 'евро', r'кап.', 'новы', 'идеальн','полн','современн']
    elite_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    elite_rep.name='elite_rep'
    df = pd.concat([df,elite_rep],axis=1)
    
    
    around_w_l=['хорош', 'косметическ']
    good_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    good_rep.name='good_rep'
    df = pd.concat([df,good_rep],axis=1)
    
    
    # df.to_csv('flats.csv', index=False)
    
    

    
    # 0.910
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep','good_rep','date_time','house_sq'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent','house_sq']
    #                                 )
    
    #0.9131909050788561 - все без date_time
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep','good_rep', 'house_sq'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent','house_sq']
    #                                 )
    
    # 0.914223255348718    - все без date_time, 'house_sq'
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep','good_rep'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                 )
    
    #  0.9145944434662227  - все без date_time, 'house_sq', 'good_rep'
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent', 'elite_rep'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                 )
    # 
    
    # 0.9145944434662227  - все без date_time, 'house_sq', 'good_rep', 'elite_rep'
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                 )
    
    # 0.9301353756001436 - все без date_time, 'house_sq', 'good_rep', 'elite_rep', dist_cent
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age']
    #                                 )
        
    

    
    
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age','dist_cent'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                 )
        
    a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                          'house_type', 'is_lst_floor'], \
                                    drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                                          'house_type']
                                    )

        
    a.df_drop_nulls()
    a.split_df_preserve_cols()


    a.ready_analyse()

    a.transform_cols()


    filt_dict, dis_ind = get_discharges(a.df, a.list_for_filtering, quantile_num=0.99)

    _, doubt_indexes = a.std_specific_disch(std_num=2)

    dis_ind = dis_ind.union(doubt_indexes)

    a.df_with_parts_drop_ind(list(dis_ind))

    # dis_ind=set()
    # dis_ind = dis_ind.union(a.df.loc[a.df['floor'] > 16].index)
    # # dis_ind = dis_ind.union(a.df.loc[a.df['total_square'] > 110].index)
    # # dis_ind = dis_ind.union(a.df.loc[a.df['total_floors'] > 17].index)
    # # dis_ind = dis_ind.union(a.df.loc[a.df['rooms_num'] > 5].index)
    # dis_ind = dis_ind.union(a.df[a.df['cost'] > 4500000].index)
    # # dis_ind = dis_ind.union(a.df[a.df['cost'] < 1000000].index)
    # a.df_with_parts_drop_ind(dis_ind)

    # # l = a.df['rooms_num'].value_counts()
    # # hist = a.df['total_square'].hist(bins=30)
    # #
    # #
    # # a.analyse_1steps(quantile_num=0.99, std_num=2)
    # #
    # #
    # #
    # # df_orig = a.df.copy()

    # # bins = [1500000,2000000,2500000,3000000, 3500000, 4500000, 5000000,5500000, 6000000,6500000,7000000,8500000,100000000]
    bins = [1500000, 3000000, 4500000, 7000000, 100000000]
    cat = pd.cut(a.df['cost'], bins)
    a.normalize()
    a.join_parts()
    categ_fr = pd.get_dummies(cat)
    a.df = pd.merge(a.df,categ_fr,left_index=True,right_index=True)
    # # y_dif=(a.df['cost']>0.5).astype(np.float32)
    # # a.df['y_dif']=y_dif

    x = a.df.drop(['cost'], axis=1)
    y = a.df['cost'].copy()
    
    # # poly = PolynomialFeatures(2)
    # # x2 = x.iloc[:, :4]
    # # x2 = poly.fit_transform(x2)
    # # x = np.concatenate([x2,x.iloc[:, 4:]],axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # pca = PCA(0.95)
    # x_train=pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)

    # reg = RandomForestRegressor()
    # # # reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.0005)
    # reg = ElasticNet(alpha=0.001)
    # # reg = LinearRegression()
    
    # reg = SVR(kernel = 'rbf',C=50,gamma=0.007, epsilon=0.1)
    # # reg = GaussianProcessRegressor()
    # reg = KNeighborsRegressor()
    # reg = LinearSVR()
    
    # grid_params=[{'n_estimators':[70,150,300], 'max_features':[4,5,9], 'max_depth':[18,25,30],'min_samples_leaf':[2,3], 'max_leaf_nodes':[200,400]}]
    # # grid_params=[{'n_estimators':[150, 200, 250],'min_samples_leaf':[2,3], 'max_leaf_nodes':[2000,2500] }]
    #
    # # grid_params=[{'min_samples_split':[5,10,20,30], 'max_features':[1,3], 'max_depth':[30,40,50,60,80],'max_leaf_nodes':[200,300,400,500],'min_samples_leaf':[3,5,8,10]}]
    # grid_params = [{'C': np.linspace(10, 20, 5), 'loss':['epsilon_insensitive', 'squared_epsilon_insensitive']}]
    # grid_params = [{'C': np.linspace(50, 70, 5), 'gamma': np.linspace(0.005, 0.01, 5)}]
    # # # grid_params = [{'n_neighbors': range(5, 15, 5)}]
    # reg = get_regr_grid_estim(reg,grid_params,x_train,y_train)
    #
    #
    #
    reg = RandomForestRegressor(n_estimators=70, min_samples_leaf=2, max_features=9,max_leaf_nodes=400,max_depth=18)
    # reg = RandomForestRegressor(n_estimators=70, min_samples_leaf=2, max_features=4,max_leaf_nodes=2000,max_depth=20)


    # # reg = RandomForestRegressor(n_estimators=70, min_samples_leaf=2, max_features=4,max_leaf_nodes=2000,max_depth=20)

    # reg.fit(x_train, y_train)
    # # score_test = reg.score(x_test, y_test)
    # # score_cross_v = cross_val_score(reg, x, y, cv=5)
    # # # score_m = np.mean(score_cross_v)
    # #
    # #
    # #
    # #
    # # # reg = RandomForestRegressor(max_features=4, max_depth = 20, max_leaf_nodes = 100)
    
    kf = KFold()
    score_l= []
    for tr_inds, ts_inds in  kf.split(x):    
        reg.fit(x.iloc[tr_inds], y[tr_inds])
        score = reg.score(x.iloc[ts_inds],y[ts_inds])
        score_l.append(score)
    score_mean = np.mean(score_l)    
    draw_learning_curve(reg,x,y, np.linspace(0.1,1.0,10))
    
    
    # # draw_validation_curve(reg, a.df.drop(['cost'], axis=1),a.df['cost'], param_name='max_features',
    # #                     param_range=[4,6,8,10,12])
    # # draw_validation_curve(reg, x,y, param_name='n_estimators',param_range=[70,150, 200, 300])
    
    predicted = reg.predict(x_test)
    rmse = np.sqrt(mean_squared_error(predicted, y_test))
    dif = rmse*a.df_std['cost']
    plt.scatter(y_test,y_test-predicted)
    plt.xlabel('рыночная цена')
    plt.ylabel('величина остатков (рыночная цена - предсказанная )')
    plt.grid(b=True)
    plt.show()
    # # y_test_orig = a.de_normalize(y_test,'cost')
    # # y_predicted = a.de_normalize(predicted,'cost')
    # # y_orig = df_orig.iloc[y_test_orig.index]
    # # x = a.df.drop(['cost'], axis=1)
    # # y = a.df['cost'].copy()

    # # # draw_validation_curve(reg,a.df.drop(['cost'], axis=1),a.df['cost'], param_name='l1_ratio', param_range=[.1,.2,.3,.4,.5,.6,.7,.8,.9])



