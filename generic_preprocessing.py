"""
@author: Aayush Agrawal
@Purpose - Re-usable code in Python 3 for general data exploration and pre-processing in modeling process
"""

## Importing required libraries
import pandas as pd ## For DataFrame operation
import numpy as np ## Numerical python for matrix operations
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler ## Preprocessing function
import pandas_profiling ## For easy profiling of pandas DataFrame
import missingno as msno ## Missing value co-occurance analysis

####### Data Exploration ############

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
    counter = 0
    for i in df.columns:
        x = df.loc[:,i].unique()
        print(counter,i,type(df.loc[0,i]), len(x), x[0:5])
        counter +=1
        
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

####### Basic helper function ############

def join_df(left, right, left_on, right_on=None, method='left'):
    '''
    Function to outer joins of pandas dataframe
    Required Input - 
        - left = Pandas DataFrame 1
        - right = Pandas DataFrame 2
        - left_on = Fields in DataFrame 1 to merge on
        - right_on = Fields in DataFrame 2 to merge with left_on fields of Dataframe 1
        - method = Type of join
    Expected Output -
        - Pandas dataframe with dropped no variation columns
    '''
    if right_on is None:
        right_on = left_on
    return left.merge(right, 
                      how=method, 
                      left_on=left_on, 
                      right_on=right_on, 
                      suffixes=("","_y"))
    
####### Pre-processing ############    

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

def treat_missing_numeric(df,columns,how = 'mean'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mean', 'mode', 'median','ffill', numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mean':
        for i in columns:
            print("Filling missing values with mean for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mean())
            
    elif how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode())
    
    elif how == 'median':
        for i in columns:
            print("Filling missing values with median for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].median())
    
    elif how == 'ffill':
        for i in columns:
            print("Filling missing values with forward fill for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(method ='ffill')
    
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    else:
        print("Missing value fill cannot be completed")
    return df

def treat_missing_categorical(df,columns,how = 'mode'):
    '''
    Function to treat missing values in numeric columns
    Required Input - 
        - df = Pandas DataFrame
        - columns = List input of all the columns need to be imputed
        - how = valid values are 'mode', any string or numeric value
    Expected Output -
        - Pandas dataframe with imputed missing value in mentioned columns
    '''
    if how == 'mode':
        for i in columns:
            print("Filling missing values with mode for columns - {0}".format(i))
            df.ix[:,i] = df.ix[:,i].fillna(df.ix[:,i].mode()[0])
    elif type(how) == str:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(how)
    elif type(how) == int or type(how) == float:
        for i in columns:
            print("Filling missing values with {0} for columns - {1}".format(how,i))
            df.ix[:,i] = df.ix[:,i].fillna(str(how))
    else:
        print("Missing value fill cannot be completed")
    return df
    
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
        - le_dict = Dictionary of all the column and their label encoders
    '''
    le_dict = {}
    for c in columns:
        print("Label encoding column - {0}".format(c))
        lbl = LabelEncoder()
        lbl.fit(list(df[c].values.astype('str')))
        df[c] = lbl.transform(list(df[c].values.astype('str')))
        le_dict[c] = lbl
    return df, le_dict

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

####### Feature Engineering ############
def create_date_features(df,column, date_format = None, more_features = False, time_features = False):
    '''
    Function to extract date features
    Required Input - 
        - df = Pandas DataFrame
        - date_format = Date parsing format
        - columns = Columns name containing date field
        - more_features = To get more feature extracted
        - time_features = To extract hour from datetime field
    Expected Output -
        - df = Pandas DataFrame with additional extracted date features
    '''
    if date_format is None:
        df.loc[:,column] = pd.to_datetime(df.loc[:,column])
    else:
        df.loc[:,column] = pd.to_datetime(df.loc[:,column],format = date_format)
    df.loc[:,column+'_Year'] = df.loc[:,column].dt.year
    df.loc[:,column+'_Month'] = df.loc[:,column].dt.month.astype('uint8')
    df.loc[:,column+'_Week'] = df.loc[:,column].dt.week.astype('uint8')
    df.loc[:,column+'_Day'] = df.loc[:,column].dt.day.astype('uint8')
    
    if more_features:
        df.loc[:,column+'_Quarter'] = df.loc[:,column].dt.quarter.astype('uint8')
        df.loc[:,column+'_DayOfWeek'] = df.loc[:,column].dt.dayofweek.astype('uint8')
        df.loc[:,column+'_DayOfYear'] = df.loc[:,column].dt.dayofyear
        
    if time_features:
        df.loc[:,column+'_Hour'] = df.loc[:,column].dt.hour.astype('uint8')
    return df

def target_encoder(train_df, col_name, target_name, test_df = None, how='mean'):
    '''
    Function to do target encoding
    Required Input - 
        - train_df = Training Pandas Dataframe
        - test_df = Testing Pandas Dataframe
        - col_name = Name of the columns of the source variable
        - target_name = Name of the columns of target variable
        - how = 'mean' default but can also be 'count'
	Expected Output - 
		- train_df = Training dataframe with added encoded features
		- test_df = Testing dataframe with added encoded features
    '''
    aggregate_data = train_df.groupby(col_name)[target_name] \
                    .agg([how]) \
                    .reset_index() \
                    .rename(columns={how: col_name+'_'+target_name+'_'+how})
    if test_df is None:
        return join_df(train_df,aggregate_data,left_on = col_name)
    else:
        return join_df(train_df,aggregate_data,left_on = col_name), join_df(test_df,aggregate_data,left_on = col_name)