
def df_drop(df,list_indexes=[],na=False,nn=False,stb='',list_duples=[]):

    if list_indexes:
        df = df.drop(list_indexes)
    elif na==True:
        df = df.dropna()
    elif nn==True and stb:
        df = df.loc[df[stb].notnull()]
    else:
        if list_duples: df = df.drop_duplicates(list_duples)
        else:  list_duples: df = df.drop_duplicates()

    df = df.reset_index()
    df = df.drop('index', axis=1)

    return df