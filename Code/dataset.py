# CS760 - Machine Learning
# Final Project - No Show Prediction
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu
# Spring 2019
#=============================================================
""" Load Dataset """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


_FILE_PATHS = {
    'test': 'Data/Test_Data_Set.csv',
    'train': 'Data/Train_Data_Set.csv',
    'merged': 'Data/Data.csv'
}

_NUM_EXAMPLES = {
    'total': 196666,
}

# load multiple csv files into one DataFrame
def import_data_df(file_paths):
    frames = []
    for path in file_paths:
        frames.append( pd.read_csv(path) )
    return pd.concat(frames, ignore_index=True)

# plot no-show counts for continuous variable
def plot_noshow_count(df, feature): 
    data_df.groupby([feature,'NO_SHOW'])['NO_SHOW']\
        .size().unstack().plot(kind='line',stacked=False)
    plt.show()

# bar plot no-show counts
def bar_plot_noshow_count(df, feature): 
    data_df.groupby([feature,'NO_SHOW'])['NO_SHOW']\
        .size().unstack().plot(kind='bar',stacked=False)
    plt.show()

# one hot encoding to integer
# columns = list of strings with columns to be converted
# integers = list of integers corresponding to columns 
def ohe_to_int(df, columns, integers, feature_name):
    # add new column
    df.loc[:,feature_name] = pd.Series(np.zeros(df.shape[0]).astype(int), index=df.index)
    # undo one-hot encoding
    i = 0
    for col in columns:
        df[feature_name] += df[col]*integers[i]
        i+=1
        # drop on-hot encoding columns
        df.__delitem__(col)
    return df
'''
def ohe_to_int_slow(df, columns, integers, feature_name):
    # add new column
    df.loc[:,feature_name] = pd.Series(np.zeros(df.shape[0]).astype(int), index=df.index)
    for index, row in df.iterrows():
        i = 0
        for col in columns:
            if row[col] == 1:
                df.loc[index,feature_name] = integers[i]
                break
            i+=1
        if index%500 == 0: print(index)
    return df
'''




if __name__ == "__main__":
    
    # load data
    paths = [_FILE_PATHS['merged']]
    data_df = import_data_df(file_paths=paths)


    noShow_X, noShow_y = data_df.iloc[:, :-1], data_df.iloc[:, -1]

    print(noShow_X)
    print(noShow_y)




    # data_df.to_csv('Data/Data.csv', index=False)


'''
    # print data info
    for i in data_df.columns.values:
        print("'"+i+"'", end=", ")
    #print(data_df.info(verbose=True))
    #print(data_df.describe())

    # undo one-hot encoding for APPT_HOUR
    columns = ['APPT_HOUR_8', 'APPT_HOUR_9', 'APPT_HOUR_10', 'APPT_HOUR_11',\
        'APPT_HOUR_13', 'APPT_HOUR_14', 'APPT_HOUR_15', 'APPT_HOUR_16',\
        'APPT_HOUR_17', 'APPT_HOUR_18', 'APPT_HOUR_19', 'APPT_HOUR_20']
    integers = list(range(8,21)).remove(12)
    data_df_int = ohe_to_int(data_df, columns, integers, 'APPT_HOUR')

    # undo one-hot encoding for CLINIC
    columns = ['CLINIC_ONE', 'CLINIC_TWO', 'CLINIC_THREE', 'CLINIC_FOUR',\
        'CLINIC_FIVE', 'CLINIC_SIX', 'CLINIC_SEVEN', 'CLINIC_EIGHT',\
        'CLINIC_NINE', 'CLINIC_TEN', 'CLINIC_ELEVEN']
    integers = list(range(1,12))
    data_df_int = ohe_to_int(data_df, columns, integers, 'CLINIC')

    
    #plot_noshow_count(data_df_int, 'PERCENT_NO_SHOWS_LAST_12_MONTHS_STANDARDIZED')
    

    '''
