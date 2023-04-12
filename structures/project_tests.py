import pandas as pd
import os
import shutil
from importlib import reload
import avito.avito_flat_parser

from datetime import date, datetime,timedelta



def split_df(filename):


    df = pd.read_csv(filename)
    df=df.drop_duplicates()
    sizes = []


    df_flat_age = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\flats_age.csv',sep=';')
    dict_house_age=avito.avito_flat_parser.AvitoFlatParser.get_house_age(df_flat_age)


    if not os.path.exists('tests'):
        os.mkdir('tests')

    if not os.path.exists('tests/1'):
        os.mkdir('tests/1')
        os.mkdir('tests/1/ark')
        df1=df.iloc[range(100,110)]
        df2 = df.iloc[range(70, 80)]
        df2.to_csv('tests/1/'+'flats.csv', index=False)
        df3 = df.iloc[range(110, 120)]
        df3.to_csv('tests/1/'+'flats_add.csv', index=False)
        digest_df(df1,'tests/1/', sizes,dict_house_age)


    fin_df = pd.read_csv('tests/1/'+'flats.csv')
    fin_df = fin_df.drop_duplicates()
    if len(fin_df)==30 and not os.path.exists('tests/1/'+'flats_add.csv') and (sizes[0],sizes[1])==(10,30):
            print('тест 1 пройден')
    sizes.clear()

    if not os.path.exists('tests/2'):
        os.mkdir('tests/2')
        os.mkdir('tests/2/ark')
        df1=[]
        df2 = df.iloc[range(110, 120)]
        df2.to_csv('tests/2/'+'flats.csv', index=False)
        df3 = df.iloc[range(120, 130)]
        df3.to_csv('tests/2/'+'flats_add.csv', index=False)
        digest_df(df1,'tests/2/', sizes,dict_house_age)


    fin_df = pd.read_csv('tests/2/'+'flats.csv')
    fin_df = fin_df.drop_duplicates()
    if len(fin_df)==20 and not os.path.exists('tests/2/'+'flats_add.csv')and (sizes[0],sizes[1])==(10,20):
            print('тест 2 пройден')
    sizes.clear()

    if not os.path.exists('tests/3'):
        os.mkdir('tests/3')
        os.mkdir('tests/3/ark')
        df1=df.iloc[range(100,110)]
        df2 = df.iloc[range(110, 120)]
        df2.to_csv('tests/3/'+'flats.csv', index=False)
        digest_df(df1,'tests/3/', sizes,dict_house_age)

    fin_df = pd.read_csv('tests/3/'+'flats.csv')
    fin_df = fin_df.drop_duplicates()
    if len(fin_df)==20 and not os.path.exists('tests/3/'+'flats_add.csv') and (sizes[0],sizes[1])==(10,20):
            print('тест 3 пройден')
    sizes.clear()

    if not os.path.exists('tests/4'):
        os.mkdir('tests/4')
        os.mkdir('tests/4/ark')
        df1=[]
        df2 = df.iloc[range(110, 120)]
        df2.to_csv('tests/4/'+'flats.csv', index=False)
        digest_df(df1,'tests/4/', sizes,dict_house_age)

    fin_df = pd.read_csv('tests/4/' + 'flats.csv')
    fin_df = fin_df.drop_duplicates()
    if len(fin_df) == 10 and not os.path.exists('tests/4/' + 'flats_add.csv')and (sizes[0],sizes[1])==(10,10):
        print('тест 4 пройден')
    sizes.clear()

    if not os.path.exists('tests/5'):
        os.mkdir('tests/5')
        os.mkdir('tests/5/ark')
        df1=[]
        digest_df(df1,'tests/5/', sizes,dict_house_age)

    if (sizes[0], sizes[1]) == (0, 0):
        print('тест 5 пройден')
    sizes.clear()

    if not os.path.exists('tests/6'):
        os.mkdir('tests/6')
        os.mkdir('tests/6/ark')
        df1=df.iloc[range(100,110)]

        #df1=df1.reset_index()
        digest_df(df1,'tests/6', sizes,dict_house_age)

    fin_df = pd.read_csv('tests/6/' + 'flats.csv')
    fin_df = fin_df.drop_duplicates()
    if len(fin_df) == 10 and os.path.exists('tests/6/' + 'flats.csv') and (sizes[0], sizes[1]) == (0, 10):
        print('тест 6 пройден')

def digest_df(flats_frame, dirname, sizes,dict_house_age):
    old=os.getcwd()
    os.chdir(dirname)


    if os.path.exists('flats_add.csv'):
        old_flats_frame = pd.read_csv('flats_add.csv')
        if len(flats_frame)!=0:
            flats_frame = pd.concat([flats_frame, old_flats_frame], sort=False, ignore_index=True)
        else:
            flats_frame = old_flats_frame
        os.remove('flats_add.csv')

    old_size = 0
    new_size = 0
    if os.path.exists('flats.csv'):
        shutil.copy('flats.csv', 'ark/flats {}.csv'.format(date.today().isoformat()))
        old_flats_frame = pd.read_csv('flats.csv')
        old_size = len(old_flats_frame)

        if len(flats_frame) != 0:
            flats_frame = pd.concat([flats_frame, old_flats_frame], sort=False, ignore_index=True)
        else:
            flats_frame = old_flats_frame


    if len(flats_frame)!=0:
        flats_frame=flats_frame.drop_duplicates()
        flats_frame = flats_frame.reset_index()
        flats_frame = flats_frame.drop('index', axis=1)
        new_size = len(flats_frame)

        # есть что-то новое
        if new_size!=old_size:
            avito.avito_flat_parser.AvitoFlatParser.calc_distance_center(flats_frame, 0, new_size-old_size, 'Владикавказ',\
                                                                         (43.024531, 44.682651))

            avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(flats_frame, 0, new_size-old_size, dict_house_age)

            # запуск функций от 0 до new_size-old_size
        flats_frame.to_csv('flats.csv', index=False)





    sizes.append(old_size)
    sizes.append(new_size)
    os.chdir(old)




if __name__=='__main__':


    reload(avito.avito_flat_parser)
    df = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\flats.csv')

    df_flat_age = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\flats_age.csv',sep=';')
    dict_house_age=avito.avito_flat_parser.AvitoFlatParser.get_house_age(df_flat_age)

    avito.avito_flat_parser.AvitoFlatParser.calc_distance_center(df2, 155, 335,'Владикавказ', (43.024531, 44.682651))
    avito.avito_flat_parser.AvitoFlatParser.calc_flats_age(df2, 155, 335, dict_house_age)

    #df.loc[df.duplicated(['adr', 'floor', 'cost', 'desc', 'total_square'])]
    # split_df(r'd:\work\autoanalysis\python\progs\scraper\flats.csv')
    #
    # df1 = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\tests\1\flats.csv')
    # df2 = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\tests\2\flats.csv')
    # df3 = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\tests\3\flats.csv')
    # df4 = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\tests\4\flats.csv')
    # df6 = pd.read_csv(r'd:\work\autoanalysis\python\progs\scraper\tests\6\flats.csv')


    #debug(split_df,r'd:\work\autoanalysis\python\progs\scraper\flats.csv')
    # import os.path as path
    #
    # two_up = path.abspath(path.join('flats.csv', "../.."))