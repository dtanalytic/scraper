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


def plot_decision_sq(x_cl,y_cl,clf,resolution,test_idx,le):
    
    x_points = np.arange(x_cl[:,0].min() , x_cl[:,0].max()+0.02,resolution[0])
    y_points = np.arange(x_cl[:,1].min(), x_cl[:,1].max()+0.02,resolution[1])
    
    X,Y=np.meshgrid(x_points,y_points)
    colors = ('orange', 'cyan', 'green', 'blue' , 'gray')
    markers = ('x','^' , 'o', 'v', 's')
    cmap=ListedColormap(colors[:len(np.unique(y_cl))])
    ar_poins = np.array([X.ravel(),Y.ravel()]).T
    Z = clf.predict(ar_poins)
    Z = Z.reshape(X.shape)
    plt.contourf(X,Y,Z,alpha=0.4,cmap=cmap)
    plt.xlim(X.min(),X.max())
    plt.ylim(Y.min(), Y.max())
    
    for i,cl in enumerate(np.unique(y_cl)):
        plt.scatter(x_cl[y_cl==cl,0],x_cl[y_cl==cl,1], marker=markers[i],label=str(le.inverse_transform([cl])), alpha=0.7, edgecolor='black', c=colors[i],s=150)
        
    if test_idx:
        x_text,y_test = x_cl[test_idx,:],y_cl[test_idx]
        plt.scatter(x_text[:,0],x_text[:,1], marker='o', alpha=1, edgecolor='black', c='', s=100, linewidth=1,label='контрольный набор')


def get_grid_estim(reg_clf, grid_params, x_train, y_train,clf=False):
    if not clf:
        grid_search = GridSearchCV(reg_clf, grid_params, cv=5, scoring='neg_mean_squared_error')
    else:
        grid_search = GridSearchCV(reg_clf, grid_params, cv=5)
    grid_search.fit(x_train, y_train)
    print(grid_search.best_params_)
    # feature_importances = grid_search.best_estimator_.feature_importances_
    # print('feature_importances \n{}'.format(feature_importances))
    # cv_res = grid_search.cv_results_
    # for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    #     print(np.sqrt(-mean_score), params)
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

    a.df_with_parts_drop_ind(dis_ind)


    
    

    # bins = [1500000,3000000, 4500000, 7000000, 100000000]
    # cat = pd.cut(a.df['cost'], bins)
    # # a.normalize()
    a.join_parts()
    # # a.df = a.df[['total_square','cost']]
    # categ_fr = pd.get_dummies(cat)
    # a.df = pd.merge(a.df,categ_fr,left_index=True,right_index=True)
    # # y_dif=(a.df['cost']>0.5).astype(np.float32)
    # # a.df['y_dif']=y_dif
    
    x = a.df.drop(['cost'], axis=1)
    # x = a.df.drop(['cost', 'total_square'], axis=1)
    y = a.df['cost'].copy()
    
    
    
    #
    #
    #
    reg = RandomForestRegressor(n_estimators=70, min_samples_leaf=2, max_features=4,max_leaf_nodes=2000,max_depth=20)
    # reg = RandomForestRegressor()
    # # reg = RandomForestRegressor(max_features=4, max_depth = 20, max_leaf_nodes = 100)
    # grid_params=[{'n_estimators':[50,100, 150],'min_samples_leaf':[2,3], 'max_leaf_nodes':[1500,2000,2200] }]

    # reg = get_grid_estim(reg,grid_params,x,y)
    draw_learning_curve(reg,x,y, np.linspace(0.1,1.0,10))
    # draw_validation_curve(reg, x,y, param_name='n_estimators',param_range=[1,5,10,20,50,70,100])
    
    
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # pca = PCA(0.9)
    # x_train=pca.fit_transform(x_train)
    # x_test = pca.transform(x_test)
    
    # reg_tree = DecisionTreeRegressor()
    # reg_tree.fit(x,y)
    
    
    
    
    # np.random.seed(5)
    
    # clf = DecisionTreeClassifier(max_depth=5, max_features=1, max_leaf_nodes=50,
    #                    min_samples_leaf=5, min_samples_split=30)
    # x=x[['total_square', 'elite_rep']]
    
    # y_cl=pd.cut(y,[1000000,3000000,4000000,600000000])
    # x_cl = x.drop(y_cl[y_cl.isnull()].index)
    # y_cl = y_cl.drop(y_cl[y_cl.isnull()].index)
    

    # le = LabelEncoder()
    # y_cl=le.fit_transform(y_cl)
    


    # x_cl_train, x_cl_test, y_cl_train, y_cl_test = train_test_split(x_cl, y_cl, test_size=0.2)
    # clf.fit(x_cl_train,y_cl_train)
    
    # # grid_params=[{'min_samples_split':[30,40,50], 'max_features':[1,2], 'max_depth':[3,5,7],'max_leaf_nodes':[30,40,50],'min_samples_leaf':[5,7]}]
    # # clf = get_grid_estim(clf,grid_params,x_cl_train,y_cl_train,clf=True)

    
    # # x_cl=np.vstack([x_cl_train,x_cl_test])
    # # y_cl = np.hstack([y_cl_train,y_cl_test])
    # # test_idx = range(len(x_cl_train),len(x_cl))
     
    # params = {'legend.fontsize': '15',
    #       'figure.figsize': (15, 5),
    #       'axes.labelsize': '19',
    #       'axes.titlesize':'22',
    #       'font.size':'15',
    #       'xtick.labelsize':'15',
    #       'ytick.labelsize':'15'}
    # pylab.rcParams.update(params)
    # plot_decision_sq(x_cl_test.values,y_cl_test,clf,resolution= (5,0.1),test_idx=None,le=le)
    # plt.xlabel(x.columns[0])
    # plt.ylabel(x.columns[1])
    # plt.legend(loc='upper left')
      
    # dot_data = export_graphviz(clf,out_file=None,rounded=True, filled=True,feature_names=x.columns)
    # graph = graph_from_dot_data(dot_data)
    # graph.write_png('house_tree.png')
   
    

    # r2 = np.mean(cross_val_score(reg_tree, x, y, cv=5))
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # reg_tree.fit(x_train, y_train)
    # y_pred = reg_tree.predict(x_test)
    # score_test = mean_squared_error(y_test,reg_tree.predict(x_test)) 
    
    
    # dot_data = export_graphviz(reg_tree,out_file=None,rounded=True, filled=True,feature_names=x.columns)
    # graph = graph_from_dot_data(dot_data)
    # graph.write_png('house_tree.png')
    
    # grid_params=[{'min_samples_split':[5,10,20,30], 'max_features':[1,3], 'max_depth':[30,40,50,60,80],'max_leaf_nodes':[200,300,400,500],'min_samples_leaf':[3,5,8,10]}]


    
    
    # y_cl = y_cl.drop(x_cl[x_cl['build_age']<1930].index)
    # x_cl = x_cl.drop(x_cl[x_cl['build_age']<1930].index) 

    # # num_samples=400
    # # ind_set=set()
    # # for val in range(len(y_cl.value_counts())):
    # #     ind_rich=y_cl[y_cl==y_cl.value_counts().index[val]][:num_samples].index
    # #     ind_set = ind_set.union(ind_rich)
    # # x_cl = x_cl.loc[list(ind_set)]
    # # y_cl = y_cl.loc[list(ind_set)]
    