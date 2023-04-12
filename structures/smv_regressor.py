import pandas as pd
from my_util.df_util import df_drop
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
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
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder


def get_grid_estim(reg_clf, grid_params, x_train, y_train,clf=False):
    if not clf:
        grid_search = GridSearchCV(reg_clf, grid_params, cv=5, scoring='neg_mean_squared_error')
    else:
        grid_search = GridSearchCV(reg_clf, grid_params, cv=5)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)


    reg_clf = grid_search.best_estimator_
    return reg_clf

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
        
        
    a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age','dist_cent'],\
                                      drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
                                    )

    # a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                       'house_type', 'is_lst_floor', 'elite_rep', 'good_rep','build_age','house_sq'], \
    #                                 drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                                       'house_type','build_age','house_sq']
    #                                 )

                                

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
    # dis_ind = dis_ind.union(a.df.loc[a.df['total_square'] > 110].index)
    # dis_ind = dis_ind.union(a.df.loc[a.df['total_floors'] > 17].index)
    # dis_ind = dis_ind.union(a.df.loc[a.df['rooms_num'] > 5].index)
    # dis_ind = dis_ind.union(a.df[a.df['cost'] > 4500000].index)
    # dis_ind = dis_ind.union(a.df[a.df['cost'] < 1000000].index)
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
    # x = a.df.drop(['cost', 'total_square'], axis=1)
    y = a.df['cost'].copy()
    
    # grid_params = [{'C': np.logspace(-3,3,4), 'gamma':np.logspace(-3,3,4),'epsilon': np.logspace(-4,2,4)}]
    # grid_params = [{'C': np.linspace(5,50,4), 'gamma':np.linspace(0.05,0.5,4),'epsilon': np.linspace(0.005,0.05,4)}]

    # reg = SVR(kernel = 'rbf',C=50,gamma=0.007, epsilon=0.1)
    # {'C': 3.4000000000000004, 'epsilon': 0.35, 'gamma': 0.05}
    # reg = SVR(kernel = 'rbf',C=50,gamma=0.05, epsilon=0.05)
    # reg = SVR(kernel = 'rbf')
    # grid_params = [{'C': np.linspace(0.1,10,4), 'gamma':np.linspace(0.005,0.05,4),'epsilon': np.linspace(0.05,0.5,4)}]

    reg = LinearSVR()
    
    # l=np.logspace(-4,2,4)
    # reg = get_grid_estim(reg,grid_params,x,y)
    draw_learning_curve(reg,x,y, np.linspace(0.1,1.0,10))
    # draw_validation_curve(reg, x,y, param_name='n_estimators',param_range=[1,5,10,20,50,70,100])
    
    
    params = {'legend.fontsize': '15',
          'figure.figsize': (15, 5),
          'axes.labelsize': '19',
          'axes.titlesize':'22',
          'font.size':'15',
          'xtick.labelsize':'15',
          'ytick.labelsize':'15'}
    pylab.rcParams.update(params)
    
    # x = np.linspace(-5,5,20)
    # y = np.zeros(shape=(20,),dtype=int)
    # y[5:15]=1
    # colors=['red','blue']
    # fig, ax = plt.subplots()
    # for idx,point in enumerate(x):        
    #     ax.scatter([point],[0],color=colors[y[idx]])
        
    # ax.set_ylim([-0.1,0.1])
    # ax.yaxis.set_major_locator(plt.NullLocator())
    
    # x2=np.square(x)

    # fig,ax = plt.subplots()
    # for idx,_ in enumerate(x):
        
    #     plt.scatter(x[idx],x2[idx],color=colors[y[idx]])
    
    # ax.hlines(6.8,-5,5,color='green',linestyle='dashed')   
    
    
    
    