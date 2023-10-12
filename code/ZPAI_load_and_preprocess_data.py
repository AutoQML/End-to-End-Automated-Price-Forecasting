import re
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd

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


    ###############################################
    # SHAP part
    # prepare the date and train it on XGBRegressor 
    ###############################################

    #get features
    y = merged_df['price']
    y = y.astype('category').cat.codes
    X_cat = merged_df.drop('price', axis=1)

    #fit encoder
    enc = OneHotEncoder( categories=['object'])
    enc.fit(X_cat)

    #transform categorical features
    X_encoded = enc.transform(X_cat).toarray()

    #create feature matrix
    feature_names = X_cat.columns
    new_feature_names = enc.get_feature_names_out(feature_names)

    X = pd.DataFrame(X_encoded, columns= new_feature_names)

    print(f"Values for X:\n {X}")

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


    # # Reset the index
    # merged_df.reset_index(inplace=True)

    # ###################################
    # # use featuretools to create new or delete unneeded features
    # ###################################
    # # es = ft.EntitySet("taxi")

    # es = ft.EntitySet(id="models_entity_set")

    # # create an entity set
    # es = es.add_dataframe(
    #     dataframe_name="merged_df",
    #     dataframe=merged_df,
    #     index="index",
    # )
    # # print the entityset
    # print(f"EntytySet: \n {es['merged_df']}")
    # print(f"EntytySet: \n {es['merged_df']}")
    

    # feature_matrix, features = ft.dfs(entityset=es,
    #                            max_features  = 6,
    #                            target_dataframe_name="merged_df")
    
    # # features = ft.dfs(entityset=es,
    # #             target_dataframe_name="merged_df",
    # #             features_only=True)
    

    # print(f"Feature matrix: \n {feature_matrix}")
    # print(f"Features: \n {features}")

    # ft.selection.remove_highly_null_features(feature_matrix)

    # new_fm, new_features = remove_single_value_features(feature_matrix, features=features)
    # # print(f"NEW Feature matrix: \n {new_fm}")

    # # feature_matrix_enc, features_enc = ft.encode_features(feature_matrix, features)
    # # print(feature_matrix_enc)

    # merged_df[columns_to_replace] = merged_df[columns_to_replace].fillna(0)


    return(merged_df)
