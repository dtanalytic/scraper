import pandas as pd
from my_util.df_util import df_drop

import numpy as np
from structures.df_analyser import DfFlatsPrepareSteps
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score





if __name__=='__main__':
    df = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats.csv')

    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'adr'])
    a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent'],\
                                      drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
                                    )

    # a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                       'house_type', 'is_lst_floor'], \
    #                                 drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
    #                                                       'house_type']
    #                                 )

    # df_show=df.drop(['desc','metro','m_distance','adr','dist_cent','page_num','cl_adr'],axis=1)
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
    # # dis_ind = dis_ind.union(a.df.loc[a.df['total_square'] > 110].index)
    # # dis_ind = dis_ind.union(a.df.loc[a.df['total_floors'] > 17].index)
    # # dis_ind = dis_ind.union(a.df.loc[a.df['rooms_num'] > 5].index)
    # dis_ind = dis_ind.union(a.df[a.df['cost'] > 4500000].index)
    # # dis_ind = dis_ind.union(a.df[a.df['cost'] < 1000000].index)
    a.df_with_parts_drop_ind(dis_ind)

    # # a.analyse_1steps(quantile_num=0.99, std_num=2)

    # df_orig = a.df.copy()

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
    
    
    
    # poly = PolynomialFeatures(3)
    # x2 = x.iloc[:, :6]
    # x2 = poly.fit_transform(x2)
    # x = np.concatenate([x2,x.iloc[:, 6:]],axis=1)



    reg = ElasticNet(alpha=0.001,fit_intercept=True)
    reg.fit(x,y)
    r2 = np.mean(cross_val_score(reg, x, y, cv=5))
    
    # draw_learning_curve(reg,x,y, np.linspace(0.1,1.0,10))
    # draw_validation_curve(reg, a.df.drop(['cost'], axis=1),a.df['cost'], param_name='alpha',
    #                     param_range=np.linspace(0.00000001,0.01,20))
    
    grid_params=[{'alpha':np.linspace(0.0000001,0.0000008,10)}]

    
    
    
    
    # np.random.seed(0)
    # inds = np.random.choice(len(x),size=1000,replace=False)
    
    # x_pl = x.iloc[inds]['total_square'].to_frame()
    # y_pl = y.iloc[inds]
    # x_pl = x['total_square'].to_frame()
    # y_pl = y
    # reg.fit(x_pl,y_pl)
    
    # r2 = np.mean(cross_val_score(reg, x_pl, y_pl, cv=5))
    
    # b0=reg.intercept_
    # b1 = reg.coef_[0]
    
    # n=2
    
    # df = pd.merge(x_pl,y_pl,left_index=True, right_index=True)
    # sc = StandardScaler()    
    # df = sc.fit_transform(df)  
    
    # dists = distance.pdist(df, 'euclidean')
    # dists = distance.squareform(dists)
    # dists2 = dists.copy()
    # dists2.sort(axis=1)
    
    # inds=dists2[:,1].argsort(axis=0)[-len(dists2)//40:]
    
    # inds=inds[[0,-10]]
    


    # fig, ax = plt.subplots(figsize=(12,4))
    # import matplotlib.pylab as pylab

    # params = {'legend.fontsize': '15',
    #       'figure.figsize': (15, 5),
    #       'axes.labelsize': '19',
    #       'axes.titlesize':'22',
    #       'font.size':'15',
    #       'xtick.labelsize':'15',
    #       'ytick.labelsize':'15'}
    # pylab.rcParams.update(params)
    # pylab.rcParams.update(params)
    # ax.scatter(x_pl,y_pl, c='steelblue')
    # ax.plot(x_pl,reg.predict(x_pl),color='black', label='y={}*x+{}'.format(int(b1),int(b0)))
    # ax.set_xlabel('площадь квартиры')
    # ax.set_ylabel('цена квартиры')
    # ax.set_xlim(0,250)
    # ax.set_ylim(0,10500000)
    # ax.set_yticks([2000000,4000000,6000000,8000000])
    # ax.legend(loc='lower right')
    # ax.set_title('зависимость цены квартиры от площади')
    
    # # ind = y_pl[y_pl==np.max(y_pl)].index[0]
    # # ind=inds
    # colors=['lightgreen','orange','cyan']
    # for j,ind in enumerate(inds):
        
    #     ax.hlines(y_pl.iloc[ind],0,x_pl.iloc[[ind]],color=colors[j%len(colors)],linestyle='dashed')
    #     ax.vlines(x_pl.iloc[ind].values[0],0,max(y_pl.iloc[[ind]].values[0],reg.predict(x_pl.iloc[[ind]])[0]),color=colors[j%len(colors)],linestyle='dashed')    
    #     ax.hlines(reg.predict(x_pl.iloc[[ind]]),0,x_pl.iloc[[ind]],color=colors[j%len(colors)],linestyle='dashed')
        
    #     ax.scatter(x_pl.iloc[[ind]], y_pl.iloc[ind], color='red')
        
    #     ax.text(x_pl.iloc[[ind]].values[0][0]+1, y_pl.iloc[ind]-200000,'x'+str(j))
        
        
    #     style_text = dict(size=10, color  = 'black',fontsize='15')
    #     ax.text(2,y_pl.iloc[ind]+10000,' {} млн - истинная стоимость точки - {}'.format(y_pl.iloc[ind]/1000000, 'x'+str(j)), **style_text)
    #     ax.text(2,reg.predict(x_pl.iloc[[ind]])+10000,'{} млн - предсказанная стоимость точки {}'.format(round(reg.predict(x_pl.iloc[[ind]])[0]/1000000,2), 'x'+str(j))
    #                 , **style_text)



        # ax.hlines(y_pl[ind],0,x_pl.loc[[ind]],color='green',linestyle='dashed')
        # ax.vlines(x_pl.loc[ind],0,y_pl[[ind]],color='green',linestyle='dashed')    
        # ax.hlines(reg.predict(x_pl.loc[[ind]]),0,x_pl.loc[[ind]],color='green',linestyle='dashed')
        
        # ax.scatter(x_pl.loc[[ind]], y_pl[ind], color='red')
        
        # ax.text(x_pl.loc[[ind]].values[0][0]+1, y_pl[ind]-200000,'x1')
    
        
        # style_text = dict(size=10, color  = 'black')
        # ax.text(2,y_pl[ind]+10000,' {} млн - истинная стоимость точки - {}'.format(y_pl[ind]/1000000, 'x1'), **style_text)
        # ax.text(2,reg.predict(x_pl.loc[[ind]])+10000,'{} млн - предсказанная стоимость точки {}'.format(round(reg.predict(x_pl.loc[[ind]])[0]/1000000,2), 'x1')
        #             , **style_text)

