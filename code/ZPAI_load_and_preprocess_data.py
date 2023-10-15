import re
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ZPAI_prepare_data_for_ml import prepare_data_for_ml
from ZPAI_common_functions import load_csv_data, create_path, read_yaml

import featuretools as ft

import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn.metrics import accuracy_score,confusion_matrix, mean_absolute_percentage_error

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn import tree

# import xgboost as xgb

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)

# import featuretools as ft


def load_and_preprocess_data(datasets: list,
                            config: dict) -> None:
    """
    Summary.

    Detailed description

    Parameters
    ----------
    machine_model : str
        Description of arg1
    GLOBAL_TXT_SUMMARY_FILE : str
        Description of arg2
    GLOBAL_YAML_SUMMARY_FILE : str
        Description
    config : dict
        Description
    
    Returns
    -------


    """
    PROCESSING_TYPE = 0
    # es = ft.EntitySet(id="my_entity_set")

    REPO_PATH = config["general"]["repo_path"]

    dataframe_list = []

    #################################
    # load the datasets and store them within a list
    ################################
    for dataset in datasets:

        ########################
        # extract model name
        ########################
        # # Define a pattern to match the part you want to delete
        # pattern = 'Caterpillar-'
        # # Use re.sub() to replace the matched pattern with an empty string
        # model_name = re.sub(pattern, '', dataset)
        # print(f"Model-1 name: {model_name}")
        # extract model name
        DELETE_BRAND_STR = 'Caterpillar-'
        model_name = str(dataset).replace(DELETE_BRAND_STR, "")
        print(f"Model name: {model_name}")


        dataframe_list.append(dataset)

        # set path to CSV data files 
        FILE_PATH_IN = Path(REPO_PATH, 'data', dataset) 

        # Get all CSV files for the current dataset
        list_of_files = FILE_PATH_IN.glob('*.csv')

        # get the most up-to-date file
        latest_file = max(list_of_files,  key=lambda x: x.stat().st_mtime)
        # print('Choosen file: ', latest_file)

        # Confert file path to string
        latest_file_string = str(latest_file)
        # print('String of latest file: ', latest_file_string)

        # searching string
        match_str = re.search(r'\d{4}-\d{2}-\d{2}', latest_file_string)
        
        # feeding format
        file_creation_date = datetime.strptime(match_str.group(), '%Y-%m-%d').date()
        
        # printing result
        # print("Date of input file : " + str(file_creation_date))

        # get most actual input data file (csv file)
        working_file = latest_file

        # load the data
        dataset_df = load_csv_data(working_file)

            
        # Delete unnamed / index column
        if set(['Unnamed: 0']).issubset(dataset_df.columns):
            dataset_df = dataset_df.drop('Unnamed: 0', axis=1)

        # add model column
        dataset_df['model'] = model_name

        # assing new dataframe to appropriate / last dataset name in the list
        dataframe_list[-1] = dataset_df


    ################################
    # merge all datasets into one dataset
    ###############################
    for i in range(len(dataframe_list)-1):
        if i == 0:
            merged_df = pd.concat([dataframe_list[i], dataframe_list[i+1]], axis=0, ignore_index=True)
        else:
            merged_df = pd.concat([merged_df, dataframe_list[i+1]], axis=0, ignore_index=True)

    # print(merged_df)

    # Save the DataFrame as a CSV file with the NaN values
    merged_df.to_csv("./data/merged-files/merged-files-final-2023-10-15-NaN.csv", index=False)
    
    ##############################
    # replace NaN values by 0 for all 
    ##############################
    # Get the current column names
    column_names = merged_df.columns.tolist()

    # Specify the names/columns with no replacement
    names_to_delete = ['model', 'extension', 'location', 'working_hours', 'const_year', 'price']

    # Create a list of column where the NaN values should be replaced by 0
    columns_to_replace = [col for col in column_names if col not in names_to_delete]
    # print(f"Column names to replace: {columns_to_replace}")

    # Replace NaN values with zeros
    merged_df[columns_to_replace] = merged_df[columns_to_replace].fillna(0)

    print(merged_df.info())

    # Save the DataFrame as a CSV file
    merged_df.to_csv("./data/merged-files/merged-files-final-2023-10-14.csv", index=False)

    # ###################################
    # # extract the label
    # ##################################

    # X_cat = merged_df.drop('price', axis=1)
    # y = merged_df['price'].copy()

    # #fit encoder
    # enc = OneHotEncoder()
    # enc.fit(X_cat)

    # #transform categorical features
    # X_encoded = enc.transform(X_cat).toarray()

    # #create feature matrix
    # feature_names = X_cat.columns
    # new_feature_names = enc.get_feature_names_out(feature_names)

    # X = pd.DataFrame(X_encoded, columns= new_feature_names)

    # # X

    # model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)


    # model.fit(X, y)

    # #Get predictions
    # y_pred = model.predict(X)

    # # print(confusion_matrix(y, y_pred))
    # # accuracy_score(y, y_pred)

    # #Get predictions
    # print(mean_absolute_percentage_error(y, y_pred))




    if PROCESSING_TYPE == 1:
        ###################################
        # extract the label
        ##################################

        X_train = merged_df.drop('price', axis=1)
        y_train = merged_df['price'].copy()

        ##################################
        # get the original feature names
        ##################################
        X_train_feature_names = X_train.columns

        #################################
        # One-hot encode categorical features
        #################################

        # define categorical features
        cat_features = ['model', 'extension', 'location']

        # separate numerical and categorical columns
        df_cat = X_train[cat_features].copy()
        df_num = X_train.drop(cat_features, axis=1)

        # get the column names
        cat_feature_names = df_cat.columns
        num_feature_names = df_num.columns

        #fit encoder
        enc = OneHotEncoder()
        enc.fit(df_cat)

        #transform categorical features
        X_encoded = enc.transform(df_cat).toarray()

        #create feature matrix with one-hot-encoded feature/column names
        new_cat_feature_names = enc.get_feature_names_out(cat_feature_names)

        df_cat_encoded = pd.DataFrame(X_encoded, columns= new_cat_feature_names)

        print(f"Values for df_cat_encoded:\n {df_cat_encoded}")

        #####################################
        # Standard scale the encoded and unencoded features
        #####################################

        # Initialize the StandardScaler for df_cat_encoded
        scaler_df_cat_encoded = StandardScaler()

        # Fit and transform the data
        scaled_data = scaler_df_cat_encoded.fit_transform(df_cat_encoded)

        # Convert the scaled data back to a dataframe
        scaled_df_cat = pd.DataFrame(scaled_data, columns=new_cat_feature_names)

        print(f"Values for scaled_df_cat:\n {scaled_df_cat}")


        # Initialize the StandardScaler for df_num
        scaler_df_num = StandardScaler()

        # Fit and transform the data
        scaled_data = scaler_df_num.fit_transform(df_num)

        # Convert the scaled data back to a dataframe
        scaled_df_num = pd.DataFrame(scaled_data, columns=num_feature_names)

        print(f"Values for scaled_df_num:\n {scaled_df_num}")

        #########################################
        # merge the encoded & unencoded data
        #########################################

        X_train_concat = pd.concat([scaled_df_cat, scaled_df_num], axis=1)

        print(f"Values for X_train_concat:\n {X_train_concat}")


        ##########################################
        #
        ##########################################

        # Get the number of columns
        num_columns = X_train_concat.shape[1]

        print(f'The DataFrame has {num_columns} columns.')


        model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
        # Fit the Model
        model.fit(X_train_concat, y_train)

        #Get predictions
        y_pred = model.predict(X_train_concat)
        print(f"mean absolute percentage erro: {mean_absolute_percentage_error(y_train, y_pred)}")

        ##########################################
        # Interpret this model using SHAP values. To do this, we pass our model 
        # into the SHAP Explainer function. This creates an explainer object. 
        # We use this to calculate SHAP values for every observation in the feature matrix.
        #########################################
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train_concat)

        # shap.plots.waterfall(shap_values[0])
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
        plt.savefig('waterfall.png')

        # print(f"Shap values: {shap_values}")

        i = 4
        # shap.force_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns)
        # shap.summary_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns, show=False)
        # shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        # plt.savefig('scratch.png')
        plt.figure()
        shap.summary_plot(shap_values, features=X, feature_names=X_train_concat.columns, plot_type='bar', show=False)
        plt.savefig('barplot.png')

        #get number of unique categories for each feature 
        n_categories = []
        for feat in X_train_feature_names[:-1]:
            n = X_train[feat].nunique()
            n_categories.append(n)
            
        print("Feature categories: ", n_categories)


    if PROCESSING_TYPE == 2:
        ###################################
        # extract the label
        ##################################

        X_train = merged_df.drop('price', axis=1)
        y_train = merged_df['price'].copy()

        ##################################
        # get the original feature names
        ##################################
        X_train_feature_names = X_train.columns

        #################################
        # One-hot encode categorical features
        #################################

        #fit encoder
        enc = OneHotEncoder()
        enc.fit(X_train)

        #transform categorical features
        X_encoded = enc.transform(X_train).toarray()

        #create feature matrix with one-hot-encoded feature/column names
        new_cat_feature_names = enc.get_feature_names_out(X_train_feature_names)

        X_train_encoded = pd.DataFrame(X_encoded, columns= new_cat_feature_names)

        print(f"Values for X_train_encoded:\n {X_train_encoded}")

        #####################################
        # Standard scale the encoded and unencoded features
        #####################################

        # Initialize the StandardScaler for X_train_encoded
        scaler = StandardScaler()

        # Fit and transform the data
        scaled_data = scaler.fit_transform(X_train_encoded)

        # Convert the scaled data back to a dataframe
        X_train_scaled = pd.DataFrame(scaled_data, columns=new_cat_feature_names)

        print(f"Values for X_train_scaled:\n {X_train_scaled}")

        ##########################################
        #
        ##########################################

        # Get the number of columns
        num_columns = X_train_scaled.shape[1]
        print(f'The DataFrame has {num_columns} columns.')


        model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
        # Fit the Model
        model.fit(X_train_scaled, y_train)

        #Get predictions
        y_pred = model.predict(X_train_scaled)
        print(f"mean absolute percentage erro: {mean_absolute_percentage_error(y_train, y_pred)}")

        ##########################################
        # Interpret this model using SHAP values. To do this, we pass our model 
        # into the SHAP Explainer function. This creates an explainer object. 
        # We use this to calculate SHAP values for every observation in the feature matrix.
        # #########################################
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X_train_scaled)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_train_scaled)

        print(f"explainer.expected_value: {explainer.expected_value}")
        print(f"shap_values: {shap_values}")

        # shap.plots.waterfall(shap_values[0])
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values.values[1])
        # shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[1])
        # shap.plots._waterfall.waterfall_legacy(shap_values[1], show=False)
        plt.savefig('waterfall.png')

        #get number of unique categories for each feature 
        n_categories = []
        for feat in X_train_feature_names[:-1]:
            n = X_train[feat].nunique()
            n_categories.append(n)
            
        print("Feature categories: ", n_categories)



        new_shap_values = []
        for values in shap_values.values:
        # for values in shap_values.values:
            
            #split shap values into a list for each feature
            values_split = np.split(values , np.cumsum(n_categories))
            
            #sum values within each list
            values_sum = [sum(l) for l in values_split]
            
            new_shap_values.append(values_sum)


        print(len(new_shap_values),sum(new_shap_values[1]),sum(shap_values.values[1]))
        # print(len(new_shap_values),sum(new_shap_values[1]),sum(shap_values[1]))
        print(new_shap_values[1])


        #replace shap values
        shap_values = np.array(new_shap_values)

        #replace data with categorical feature values 
        new_data = np.array(X_train_scaled)
        shap_values.data = np.array(new_data)

        #update feature names
        # shap_values.X_train_feature_names = list(X_train_scaled.columns)


        # waterfall plot
        plt.figure()
        # shap.plots._waterfall.waterfall_legacy(shap_values[1], show=False)
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values.values[1])
        plt.savefig('waterfall_2.png')

        # # waterfall plot
        # shap.plots.waterfall(shap_values[1], show=False)

        # plt.savefig(path.format('category_shap.png'),dpi=100,bbox_inches='tight')
        



        # i = 4
        # # shap.force_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns)
        # # shap.summary_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns, show=False)
        # # shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        # # plt.savefig('scratch.png')
        # plt.figure()
        # shap.summary_plot(shap_values, features=X, feature_names=X_train_concat.columns, plot_type='bar', show=False)
        # plt.savefig('barplot.png')

        







    if PROCESSING_TYPE == 3:

        #################################################
        # convert int64 to float
        #################################################
        # List the columns you want to keep as int64
        columns_to_keep_as_objets = ['extension', 'model', 'location']

        # Iterate through the columns and convert to float if not in the keep list
        for col in merged_df.columns:
            if col not in columns_to_keep_as_objets:
                merged_df[col] = merged_df[col].astype(float)

        # Check the data types after conversion
        print(merged_df.dtypes)

        # merged_df = merged_df.astype({'working_hours': float, 'const_year': float})

        # split data into X_train and y_train
        X_train = merged_df.drop('price', axis=1)
        y_train = merged_df['price'].copy()

        num_feateres = X_train.drop(columns_to_keep_as_objets, axis=1)
        print(f"Columns with numerical features: {num_feateres}")



        num_attribs = list(num_feateres)
        cat_attribs = ['extension', 'model', 'location']

        num_pipeline = Pipeline([
            # ('imputer', SimpleImputer(strategy='median')),  # check and handle missing values
            # ('attribs_addr', CombinedAttributesAdder(add_bedrooms_per_room=True)), # add attributes
            #('attribs_adder', FunctionTransformer(add_extra_features, validate=False)), # add attributes
            ('std_scaler', StandardScaler()), # scale the data
        ])

        full_pipeline = ColumnTransformer([
                ("num", num_pipeline, num_attribs),
                ("cat", OneHotEncoder(), cat_attribs), # encode the categorical data
            ])

        X_train_prepared = full_pipeline.fit_transform(X_train)

        print(f"One-Hot-Encoded dataframe: {X_train_prepared}")

        X_train_prep = pd.DataFrame(X_train_prepared)

        print(f"X_train_prep:\n {X_train_prep}")

        ###############################################
        # SHAP part
        # prepare the date and train it on XGBRegressor 
        ###############################################

        #get features
        y = merged_df['price']
        y = y.astype('category').cat.codes
        X_cat = merged_df.drop('price', axis=1)

        #fit encoder
        enc = OneHotEncoder(dtype=float)
        enc.fit(X_cat)

        #transform categorical features
        X_encoded = enc.transform(X_cat).toarray()

        #create feature matrix
        feature_names = X_cat.columns
        new_feature_names = enc.get_feature_names_out(feature_names)

        X = pd.DataFrame(X_encoded, columns= new_feature_names)

        print(f"Values for X:\n {X}")

        # Get the number of columns
        num_columns = X.shape[1]

        print(f'The DataFrame has {num_columns} columns.')


        model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
        # Fit the Model
        model.fit(X, y)

        #Get predictions
        y_pred = model.predict(X)

        # print(f"Confusion matrix: {confusion_matrix(y, y_pred)}")
        # print(f"Accuracy score: {accuracy_score(y, y_pred)}")

        print(f"mean absolute percentage erro: {mean_absolute_percentage_error(y, y_pred)}")


        #get number of unique categories for each feature 
        n_categories = []
        for feat in feature_names[:-1]:
            n = X_cat[feat].nunique()
            n_categories.append(n)
            
        print("Feature categories: ", n_categories)



        # # Prepare X and Y 
        # X = pd.get_dummies(merged_df)
        # X.drop(['price'], inplace=True, axis=1)
        # y = merged_df['price']

        # model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
        # # Fit the Model
        # model.fit(X, y)

        # load JS visualization code to notebook
        # shap.initjs()
        ##########################################
        # Interpret this model using SHAP values. To do this, we pass our model 
        # into the SHAP Explainer function. This creates an explainer object. 
        # We use this to calculate SHAP values for every observation in the feature matrix.
        #########################################
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # shap.plots.waterfall(shap_values[0])
        plt.figure()
        shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0])
        plt.savefig('waterfall.png')

        # print(f"Shap values: {shap_values}")

        i = 4
        # shap.force_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns)
        # shap.summary_plot(explainer.expected_value, shap_values[i], features=X.iloc[i], feature_names=X.columns, show=False)
        # shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
        # plt.savefig('scratch.png')
        plt.figure()
        shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_type='bar', show=False)
        plt.savefig('barplot.png')


    return(merged_df)
