"""
Created on Sat Feb 10 11:38:16 2018
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for general pre-processing in modeling process
"""

## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler ## Preprocessing function
import pandas_profiling ## For easy profiling of pandas DataFrame
import missingno as msno ## Missing value co-occurance analysis

def print_dim(df):
    '''
    Function to print the dimensions of a given python dataframe
    Required Input -
        - df = Pandas DataFrame
    Expected Output -
        - Data size
    '''
    print("Data size: Rows-{0} Columns-{1}".format(df.shape[0],df.shape[1]))


def print_dataunique(df):
    '''
    Function to print unique information for each column in a python dataframe
    Required Input - 
        - df = Pandas DataFrame
    Expected Output -
        - Column name
        - Data type of that column
        - Number of unique values in that column
        - 5 unique values from that column
    '''
    for i in df.columns:
        x = df.loc[:,i].unique()
        print(i,type(df.loc[0,i]), len(x), x[0:5])
        
def do_data_profiling(df, filename):
    '''
    Function to do basic data profiling
    Required Input - 
        - df = Pandas DataFrame
        - filename = Path for output file with a .html extension
    Expected Output -
        - HTML file with data profiling summary
    '''
    profile = pandas_profiling.ProfileReport(df)
    profile.to_file(outputfile = filename)
    print("Data profiling done")

def missing_value_analysis(df):
    '''
    Function to do basic missing value analysis
    Required Input - 
        - df = Pandas DataFrame
    Expected Output -
        - Chart of Missing value co-occurance
        - Chart of Missing value heatmap
    '''
    msno.matrix(df)
    msno.heatmap(df)
    
def drop_allsame(df):
    '''
    Function to remove any columns which have same value all across
    Required Input - 
        - df = Pandas DataFrame
    Expected Output -
        - Pandas dataframe with dropped no variation columns
    '''
    to_drop = list()
    for i in df.columns:
        if len(df.loc[:,i].unique()) == 1:
            to_drop.append(i)
    return df.drop(to_drop,axis =1)

def min_max_scaler(df,columns):
    '''
    Function to do Min-Max scaling
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns which needs to be min-max scaled
    Expected Output -
        - df = Python DataFrame with Min-Max scaled attributes
        - scaler = Function which contains the scaling rules
    '''
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(df.loc[:,columns]))
    data.index = df.index
    data.columns = df.columns
    return data, scaler

def z_scaler(df,columns):
    '''
    Function to standardize features by removing the mean and scaling to unit variance
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns which needs to be min-max scaled
    Expected Output -
        - df = Python DataFrame with Min-Max scaled attributes
        - scaler = Function which contains the scaling rules
    '''
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(df.loc[:,columns]))
    data.index = df.index
    data.columns = df.columns
    return data, scaler
    
def label_encoder(df,columns):
    '''
    Function to label encode
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns which needs to be label encoded
    Expected Output -
        - df = Pandas DataFrame with lable encoded columns
    '''
    for c in columns:
        print("Label encoding column - {0}".format(c))
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values.astype('str')))
        df[c] = lbl.transform(list(df[c].values.astype('str')))
    return df

def one_hot_encoder(df, columns):
    '''
    Function to do one-hot encoded
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns which needs to be one-hot encoded
    Expected Output -
        - df = Pandas DataFrame with one-hot encoded columns
    '''
    for each in columns:
        print("One-Hot encoding column - {0}".format(each))
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df.drop(columns,axis = 1)