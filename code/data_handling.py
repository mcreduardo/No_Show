# CS760 - Machine Learning
# Final Project - No Show Prediction
# Eduardo Moura Cirilo Rocha, mouracirilor@wisc.edu, and group
# Spring 2019

# Data handling modules

import pandas as pd

# load multiple csv files into one DataFrame
def import_data(file_paths):
    frames = []
    for path in file_paths:
        frames.append( pd.read_csv(path) )
    return pd.concat(frames, ignore_index=True)





if __name__ == "__main__":
    
    # load data
    paths = ["Data/Test_Data_Set.csv", "Data/Train_Data_Set.csv"]
    data_df = import_data(file_paths=paths)

    print(data_df.info(verbose=True))
    print(data_df)



