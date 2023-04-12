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
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import matplotlib.pylab as pylab
from structures.df_tester import form_indic_pattern_center_w_around_w_l



if __name__=='__main__':
    df = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats.csv')

    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'adr'])
    
    around_w_l=['дорог', 'капитальн', 'шикарн', 'евро', r'кап.', 'новы', 'идеальн','полн']
    elite_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    elite_rep.name='elite_rep'
    df = pd.concat([df,elite_rep],axis=1)
    
    around_w_l=['хорош', 'косметическ']
    good_rep = form_indic_pattern_center_w_around_w_l(df['desc'],'ремонт',around_w_l)
    good_rep.name='good_rep'
    df = pd.concat([df,good_rep],axis=1)
    
    
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age','house_sq','dist_cent', 'elite_rep', 'good_rep'],\
    #                                   drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent','house_sq']
    #                                 )


    a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                          'house_type', 'is_lst_floor', 'elite_rep', 'good_rep','build_age','house_sq'], \
                                    drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                                          'house_type','build_age','house_sq']
                                    )

                                

    a.df_drop_nulls()
    a.split_df_preserve_cols()

    a.ready_analyse()

    a.transform_cols()

    filt_dict, dis_ind = get_discharges(a.df, a.list_for_filtering, quantile_num=0.99)

    _, doubt_indexes = a.std_specific_disch(std_num=2)

    dis_ind = dis_ind.union(doubt_indexes)

    a.df_with_parts_drop_ind(list(dis_ind))

    dis_ind=set()
    dis_ind = dis_ind.union(a.df.loc[a.df['floor'] > 16].index)

    a.df_with_parts_drop_ind(dis_ind)



    # bins = [1500000,3000000, 4500000, 7000000, 100000000]
    # cat = pd.cut(a.df['cost'], bins)
    a.normalize()
    a.join_parts()
    # # a.df = a.df[['total_square','cost']]
    # categ_fr = pd.get_dummies(cat)
    # a.df = pd.merge(a.df,categ_fr,left_index=True,right_index=True)
    # # y_dif=(a.df['cost']>0.5).astype(np.float32)
    # # a.df['y_dif']=y_dif
    
    x = a.df.drop(['cost'], axis=1)
    y = a.df['cost'].copy()
    
    
    # poly = PolynomialFeatures(2)
    # x2 = x.iloc[:, :6]
    # x2 = poly.fit_transform(x2)
    # x = np.concatenate([x2,x.iloc[:, 6:]],axis=1)


    reg_knn = KNeighborsRegressor(n_neighbors=6)
    reg_knn.fit(x,y)
    # r2 = np.mean(cross_val_score(reg_knn, x, y, cv=5))
    
    # draw_learning_curve(reg_knn,x,y, np.linspace(0.1,1.0,10))
    # draw_validation_curve(reg_knn, a.df.drop(['cost'], axis=1),a.df['cost'], param_name='alpha',
    #                     param_range=np.linspace(0.00000001,0.01,20))
    
    
    
    reg_el = ElasticNet(alpha=0.001,fit_intercept=True)
    reg_el.fit(x,y)
    


    r2_el = np.mean(cross_val_score(reg_el, x, y, cv=5))
    r2_knn = np.mean(cross_val_score(reg_knn, x, y, cv=5))
    
    
    # draw_learning_curve(reg_knn,x,y, np.linspace(0.1,1.0,10))
    # draw_validation_curve(reg_knn, x,y, param_name='n_neighbors',
    #                     param_range=range(1,10))
    
    
    # np.random.seed(0)
    # inds = np.random.choice(len(x),size=300,replace=False)
    
    # x_pl = x.iloc[inds]['total_square'].to_frame()
    # y_pl = y.iloc[inds]

    
    # reg_knn.fit(x_pl,y_pl)
    
    # reg_el.fit(x_pl,y_pl)
    # r2_el = np.mean(cross_val_score(reg_el, x_pl, y_pl, cv=5))
    # r2_knn = np.mean(cross_val_score(reg_knn, x_pl, y_pl, cv=5))
    # params = {'legend.fontsize': '15',
    #       'figure.figsize': (15, 5),
    #       'axes.labelsize': '19',
    #       'axes.titlesize':'22',
    #       'font.size':'15',
    #       'xtick.labelsize':'15',
    #       'ytick.labelsize':'15'}
    # pylab.rcParams.update(params)
    
    # fig, ax = plt.subplots(figsize=(12,4))
    # ax.scatter(x_pl,y_pl, c='steelblue')
    # ax.plot(x_pl,reg_el.predict(x_pl),color='black', label='ElasticNet')
    # # sq_range = np.linspace(0,2.5,30)
    # sq_range = np.linspace(20,170,30)
    # ax.step(sq_range,reg_knn.predict(sq_range[:,np.newaxis]),color='brown', label='KNeighborsRegressor')
    # ax.scatter(sq_range,[0.002]*len(sq_range),color='red')
    # # ax.scatter(x_pl.iloc[[ind]], y_pl.iloc[ind], color='red')
    
    # ax.set_xlabel('площадь квартиры')
    # ax.set_ylabel('цена квартиры')
    # # ax.set_xlim(0,2.5)
    # # ax.set_ylim(0,2.5)
    # ax.set_xlim(20,170)
    # ax.set_ylim(0,6500000)
    # ax.legend(loc='lower right')
    # ax.set_title('зависимость цены квартиры от площади')


    

    
