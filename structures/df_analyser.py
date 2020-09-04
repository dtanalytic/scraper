import re
import pandas as pd
from my_util.df_util import df_drop
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
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
from sklearn.feature_selection import RFE
from structures.df_tester import get_discharges,x_std_disch_intervals
from structures.df_tester import n_sim_obj
from structures.df_tester import get_neigh_dists_matrs
from sklearn.model_selection import cross_val_score
from structures.df_tester import draw_learning_curve,draw_validation_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

def set_trace():
	from IPython.core.debugger import Pdb
	import sys
	Pdb(color_scheme='Linux').set_trace(sys._getframe().f_back)

def debug(f, *args,**kwargs):
    from IPython.core.debugger import  Pdb
    pdb=Pdb(color_scheme='Linux')
    return pdb.runcall(f, *args,**kwargs)


def draw_dependency(predicted, real):
    plt.scatter(predicted, real)
    plt.xlabel('предсказанная цена')
    plt.ylabel('фактическая цена')
    plt.plot([real.min(), real.max()], [[real.min()], [real.max()]])
    plt.show()


def mean_cross_valid_error(y_stb, table_x, reg_method, num_classes=10):
    kf = KFold(n_splits=num_classes, shuffle=True)
    # kf.get_n_splits(df4)
    r2_errors = []
    r2_tr_errors = []
    rmse_errors=[]
    rmse_tr_errors=[]

    p = np.zeros_like(y_stb)

    for training, testing in kf.split(table_x):
        x = table_x.iloc[training]
        y = y_stb[training]

        reg_method.fit(x, y)

        p[testing] = reg_method.predict(table_x.iloc[testing])
        p[training] = reg_method.predict(table_x.iloc[training])
        rmse = np.sqrt(mean_squared_error(p[testing], y_stb[testing]))
        rmse_tr = np.sqrt(mean_squared_error(p[training], y_stb[training]))
        r2 = r2_score(y_stb[testing], p[testing])

        r2_tr = r2_score(y_stb[training], p[training])
        r2_tr_errors.append(r2_tr)
        r2_errors.append(r2)
        rmse_errors.append(rmse)
        rmse_tr_errors.append(rmse_tr)

    return np.mean(r2_errors),np.mean(rmse_errors), np.mean(r2_tr_errors),np.mean(rmse_tr_errors)


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



class DfPrepareSteps():

    @classmethod
    def from_df(cls,df,analyse_cols,drop_null_cols_list=[],filter_dis_cols_list=[], year_segm_num=4):
        obj=cls()
        obj.df=df
        obj.year_segm_num=year_segm_num
        obj.df_original = df
        obj.df_parts={}
        obj.analyse_cols=analyse_cols.copy()
        obj.ind_map_steps = []
        if drop_null_cols_list:
            obj.drop_null_cols_list = drop_null_cols_list

        else: obj.drop_null_cols_list = analyse_cols.copy()

        if filter_dis_cols_list:
            obj.filter_dis_cols_list = filter_dis_cols_list

        else:
            obj.filter_dis_cols_list = [item for item in analyse_cols.copy() if item not in ['date_time']]

        obj.configure_new_variables()

        #obj.split_df_preserve_cols()

        return obj

    def ready_analyse(self):
        self.df = self.df[self.analyse_ready_cols]
        self.clean()

    def split_df_preserve_cols(self):
        for stb in self.list_preserve:
            self.df_parts.update({stb: self.df[stb]})
            self.df = self.df.drop(stb, axis=1)

    def df_drop_ind(self,df, list_indexes,rememb_step=True):
        df = df.drop(list_indexes)

        df = df.reset_index()
        if rememb_step:
            self.ind_map_steps.append(dict((k, v) for k, v in zip(df.index.values, df['index'].values)))
        df = df.drop('index', axis=1)
        return df

    def df_with_parts_drop_ind(self, indexes):

        self.df = self.df_drop_ind(self.df,indexes)
        if self.df_parts:
            for key,_ in self.df_parts.items():
                self.df_parts[key]=self.df_drop_ind(self.df_parts[key], indexes, rememb_step=False)

    def join_parts(self):
        for key, _ in self.df_parts.items():
            self.df=self.df.join(self.df_parts[key])


    def df_drop_nulls(self):

        set_ind = set()

        for stb_name in self.drop_null_cols_list:
            set_ind_new = set(self.df[self.df[stb_name].isnull()].index)
            set_ind.update(set_ind_new)
            # set_trace()

        self.df_with_parts_drop_ind(list(set_ind))

    def analyse_1steps(self, quantile_num=0.99, std_num=2):
        self.df_drop_nulls()
        self.split_df_preserve_cols()
        self.ready_analyse()
        self.transform_cols()

        filt_dict, dis_ind = get_discharges(self.df, self.list_for_filtering, quantile_num)
        _, doubt_indexes = self.std_specific_disch(std_num)
        dis_ind = dis_ind.union(doubt_indexes)
        self.df_with_parts_drop_ind(list(dis_ind))

    def normalize(self):
        self.df_std = self.df.std()
        self.df_mean =self.df.mean()
        self.df = (self.df - self.df_mean).div(self.df_std)

    def de_normalize(self,value,stb_name):
        return value*self.df_std[stb_name]+self.df_mean[stb_name]

    def show_regression_problem_els(self,dif_pred_y, similarity_stb_list, y_stb_name='cost',k=5):
        '''
        dif_pred_y - records with max predicted distance from given y
        k - number neighbours for each record in
        similarity_stb_list - to choose neighbours distance
        y_stb_name - to compare
        :return list of dfs with el&neighbours, list of mean costs:
        '''
        df_els_neighbours_list= []
        mean_y_neighbours_list=[]
        dif_pred_y_origins = find_grand_parent_el(list(dif_pred_y.index.values), self.df_original, self.ind_map_steps)
        neigh_matr, dists_matr = get_neigh_dists_matrs(self.df[similarity_stb_list])

        for item in range(len(dif_pred_y)):

            neigh_els = n_sim_obj(self.df, neigh_matr, dists_matr, list(dif_pred_y.index.values)[item], k)

            mean_y = np.mean(self.df.iloc[neigh_els.index.values][y_stb_name]) * self.df_std[y_stb_name] + self.df_mean[
                y_stb_name]

            neigh_els_origins = find_grand_parent_el(list(neigh_els.index.values), self.df_original, self.ind_map_steps)

            neigh_els = neigh_els.reset_index()
            neigh_els = neigh_els.drop('index', axis=1)
            neigh_els_origins = neigh_els_origins.reset_index()
            neigh_els_origins = pd.concat([neigh_els_origins, neigh_els['dist']], axis=1, sort=False)
            neigh_els_origins = neigh_els_origins.set_index('index')

            neigh_els_origins_with_bad_first = pd.concat([dif_pred_y_origins.iloc[[item]], neigh_els_origins],
                                                         sort=False)

            df_els_neighbours_list.append(neigh_els_origins_with_bad_first)
            mean_y_neighbours_list.append(mean_y)


        return df_els_neighbours_list,mean_y_neighbours_list

    def analyse(self,reg_stb_name,quantile_num=0.99,std_num=2,regr_type='elastic'):

        self.analyse_1steps(quantile_num,std_num)

        # from sklearn.preprocessing import StandardScaler
        # sc= StandardScaler()
        # a.df = sc.fit_transform(a.df[['cost', 'total_square', 'live_square', 'floor','build_age', 'dist_cent']])
        #self.df = (self.df - self.df.mean()).div(self.df.std())
        self.normalize()
        self.join_parts()

        reg, r2, rmse, r2_tr, rmse_tr = start_regr(self.df,reg_stb_name,regr_type)
        predicted = reg.predict(self.df.drop([reg_stb_name], axis=1))
        draw_dependency(predicted, self.df[reg_stb_name])

        return reg,r2,rmse*self.df_std[reg_stb_name],r2_tr,rmse_tr*self.df_std[reg_stb_name]

def start_regr(df,reg_stb_name,regr_type):

    x = df.drop([reg_stb_name], axis=1)
    y = df[reg_stb_name]

    if regr_type == 'elastic':
        reg = ElasticNetCV()
    if regr_type == 'linear':
        reg = LinearRegression()

    reg.fit(x, y)
    r2, rmse, r2_tr, rmse_tr = mean_cross_valid_error(y, x, reg)
    return reg,r2, rmse, r2_tr, rmse_tr

def start_regr_twice(df,reg_stb_name,regr_type,per_chunk= 0.1):
    reg,r2, rmse, r2_tr, rmse_tr = start_regr(df,reg_stb_name,regr_type)
    predicted = reg.predict(df.drop([reg_stb_name], axis=1))

    bad_regr_num = round(per_chunk*len(df))
    dif_pred_y = np.abs(df[reg_stb_name] - predicted).sort_values()[-bad_regr_num:]
    df = df.drop(dif_pred_y.index.values)
    df = df.reset_index()
    df = df.drop('index', axis=1)

    return start_regr(df,reg_stb_name,regr_type)



class DfFlatsPrepareSteps(DfPrepareSteps):

    def configure_new_variables(self):
        # self.list_preserve =['house_type']
        potent_pres_l = ['house_type','new_build','elite_rep','good_rep','date_time']
        self.list_preserve =[item for item in potent_pres_l if item in self.analyse_cols]
        self.analyse_ready_cols = [item for item in self.analyse_cols if item not in ['house_type', 'is_lst_floor','new_build','elite_rep','good_rep','date_time']]
        if 'is_lst_floor' in self.analyse_cols:
            if 'floor' not in self.analyse_ready_cols:
                self.analyse_ready_cols.append('floor')
            if 'total_floors' not in self.analyse_ready_cols:
                self.analyse_ready_cols.append('total_floors')
        if 'live_square' in self.analyse_cols:
                if 'total_square' not in self.analyse_ready_cols:
                    self.analyse_ready_cols.append('total_square')

        self.list_for_filtering = [item for item in self.filter_dis_cols_list if item not in ['live_square','good_rep','new_build','elite_rep', 'house_type',\
                                    'is_lst_floor', 'total_floors', 'floor', 'rooms_num', 'build_age','date_time']]

    def std_specific_disch(self,std_num):
        if 'total_square' in self.analyse_cols:
            return x_std_disch_intervals(self.df, 'cost', 'total_square', mult=std_num,
                              bins=[0, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
                                    140, 150, 160, 170, 180, 190, 200, 1000])
        else: return [],set()

    def transform_cols(self):
        if 'live_square' in self.analyse_cols:
            bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
            cat = pd.cut(self.df['total_square'], bins)
            live_sq = self.df['live_square'].groupby(cat).transform(np.mean)
            self.df['live_square'] = np.where(self.df['live_square'].isnull(), live_sq, self.df['live_square'])
            if 'total_square' not in self.analyse_cols:
                self.analyse_ready_cols.remove('total_square')


        if 'is_lst_floor' in self.analyse_cols:
            is_lst_floor = np.where(self.df['total_floors'] == self.df['floor'], 1, 0)
            is_lst_floor = pd.DataFrame(is_lst_floor, columns=['is_lst_floor'])
            if 'floor' not in self.analyse_cols:
                self.analyse_ready_cols.remove('floor')
            if 'total_floors' not in self.analyse_cols:
                self.analyse_ready_cols.remove('total_floors')

            self.df_parts.update({'is_lst_floor': is_lst_floor})

        if 'house_type' in self.analyse_cols:
            flat_type = pd.get_dummies(self.df_parts['house_type'])
            self.df_parts.update({'house_type': flat_type})

        if 'date_time' in self.analyse_cols:            
            y = pd.cut(self.df_parts['date_time'],self.year_segm_num)
            year_parts = pd.get_dummies(y)
            year_parts.columns = [ str(item).split()[2] for i,item in enumerate(year_parts.columns)]
            self.df_parts.update({'date_time': year_parts})

    def makeStbsFloat(self):
            for item in list(self.df.columns):
                if type(self.df[item][0]) == str:
                # if self.df[item].dtype == np.object or self.df[item].dtype == np.string_:
                    self.df[item] = self.df[item].apply(lambda x: x.replace(' ', '') if pd.notnull(x) else x)
                self.df[item] = self.df[item].astype(np.float32)

    def clean(self):
            if 'total_square' in self.analyse_cols:
                self.df.loc[self.df['total_square'].notnull(), 'total_square'] = self.df.loc[
                    self.df['total_square'].notnull(), 'total_square'].map(
                    lambda x: x.replace('м²', ''))
            if 'live_square' in self.analyse_cols:
                self.df.loc[self.df['live_square'].notnull(), 'live_square'] = self.df.loc[self.df['live_square'].notnull(), 'live_square'].map(
                    lambda x: x.replace('м²', ''))
            if 'rooms_num' in self.analyse_cols:
                self.df['rooms_num'] = np.where(self.df.rooms_num.str.contains('студии') | self.df.rooms_num.str.contains('своб.'), '1',
                                            self.df.rooms_num.str.findall('(\d+)').str[0])

            self.makeStbsFloat()


def find_grand_parent_el(ind_el_list,df_oringin,ind_map_steps):
    ind_orig_list=[]
    ind_map_steps_rev = ind_map_steps[::-1]
    for cur_ind in ind_el_list:
        for step in ind_map_steps_rev:
            cur_ind = step[cur_ind]
        ind_orig_list.append(cur_ind)
    return df_oringin.iloc[ind_orig_list]

def add_cols_func_list(df,col_func_dict):
    for col, func in col_func_dict.items():
        df[func.__name__ + '_' + col] = func(df[col])
        return df

if __name__=='__main__':

   # from importlib import reload
   # reload(my_util.df_util)


    df = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats.csv')
    # df2 = pd.read_csv(r'c:\work\dev\python\progs\scraper\flats2.csv')
    # df_new = pd.concat([df2, df], sort=False, ignore_index=True)
    # df_new.to_csv('flats.csv', index=False)

    df = df_drop(df, list_duples=['total_square', 'total_floors', 'floor', 'rooms_num', 'house_type', 'adr'])
    # a=DfFlatsPrepareSteps.from_df(df,['cost', 'total_square', 'live_square', 'total_floors', 'floor', 'rooms_num','house_type','is_lst_floor','build_age', 'dist_cent'],\
    #                                  drop_null_cols_list =['cost', 'total_square', 'total_floors', 'floor', 'rooms_num','house_type','build_age', 'dist_cent']
    #                                )
    # a = DfFlatsPrepareSteps.from_df(df, ['cost','total_square',
    #                              'house_type', 'is_lst_floor'], \
    #                             drop_null_cols_list=['cost','total_square',
    #                                                  'house_type']
    #                             )
    a = DfFlatsPrepareSteps.from_df(df, ['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                         'house_type', 'is_lst_floor'], \
                                    drop_null_cols_list=['cost', 'total_square', 'total_floors', 'floor', 'rooms_num',
                                                         'house_type']
                                    )


    #reg,r2,rmse,r2_tr,rmse_tr = a.analyse('cost',quantile_num=0.99,std_num=2, regr_type='elastic')

    np.random.seed(7)
    reg_stb_name = 'cost'
    regr_type = 'linear'
    a.analyse_1steps(quantile_num= 0.99, std_num= 2)
    a.normalize()
    a.join_parts()


    reg = RandomForestRegressor()
    reg = SGDRegressor(n_iter=1,warm_start=True,penalty=None,learning_rate='constant',eta0=0.0005)
    x_train, x_test, y_train, y_test = train_test_split(a.df.drop(['cost'], axis=1), a.df['cost'], test_size=0.2)







    #reg,r2, rmse, r2_tr, rmse_tr = start_regr_twice(df_tr,reg_stb_name,regr_type,per_chunk=0.001)


    #
    # ind = np.arange(0, len(a.df))
    # np.random.shuffle(ind)
    # chunk = 0.1
    #
    # df_ts = a.df.iloc[ind[:round(chunk * len(a.df))]]
    # df_tr = a.df.iloc[ind[round(chunk * len(a.df)):]]
    #
    # df_ts = df_ts.reset_index()
    # df_ts = df_ts.drop('index', axis=1)
    # df_tr = df_tr.reset_index()
    # df_tr = df_tr.drop('index', axis=1)
    # reg.fit(df_tr.drop([reg_stb_name], axis=1),df_tr[reg_stb_name])
    # # reg,r2, rmse, r2_tr, rmse_tr = start_regr(df_tr,reg_stb_name,regr_type)
    # predicted = reg.predict(df_ts.drop([reg_stb_name], axis=1))
    # rmse = np.sqrt(mean_squared_error(predicted, df_ts[reg_stb_name]))
    # r2 = r2_score(df_ts[reg_stb_name], predicted)


    #
    # coefs = pd.DataFrame({'coefs':reg.coef_,'vars':a.df.iloc[:,1:].columns.values})
    #
    # x = df_ts.drop([reg_stb_name], axis=1)
    # y = df_ts[reg_stb_name]
    #
    #
    #
    #
    #
    #
    # rmse = rmse * a.df_std[reg_stb_name]
    #
    # selector = RFE(reg,n_features_to_select=3)
    # selector.fit(df_ts.drop([reg_stb_name], axis=1), df_ts[reg_stb_name])
    # chosen=selector.support_
    #draw_dependency(predicted, df_ts[reg_stb_name])


    #dif_pred_y = np.abs(a.df[reg_stb_name] - predicted).sort_values()[-280:]


    #b=a.show_regression_problem_els(dif_pred_y,similarity_stb_list=['total_square','dist_cent','build_age'], y_stb_name='cost',k=5)



#a.df = df_drop(a.df, na=True)
# Xnew = [[129, 80, 8, 8, 5, 0, 1, 0, 0, 1]]
# c_z = reg.predict(Xnew)








    # ind_map_steps=[]
    #
    # df = df.drop([1,2,3])
    #
    # df = df.reset_index()
    #
    # ind_map_steps.append(dict((k, v) for k, v in zip(df.index.values, df['index'].values)))
    #
    # df = df.drop('index', axis=1)

    # ind_el_list = [1, 2]
    # df_oringin = a.df_original
    # ind_map_steps = a.ind_map_steps
    #
    # ind_orig_list = []
    # ind_map_steps_rev = ind_map_steps[::-1]
    # for cur_ind in ind_el_list:
    #     for step in ind_map_steps_rev:
    #         cur_ind = step[cur_ind]
    #     ind_orig_list.append(cur_ind)
    # df2 = df_oringin.iloc[ind_orig_list]
