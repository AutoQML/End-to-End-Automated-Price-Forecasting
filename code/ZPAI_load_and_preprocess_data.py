import re
import yaml
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time

from ZPAI_prepare_data_for_ml import prepare_data_for_ml
from ZPAI_common_functions import load_csv_data, create_path, read_yaml, perform_outlier_detection

import featuretools as ft

from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_single_value_features,
)
import autogluon.eda.auto as auto
import shap

from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error

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

import warnings
warnings.filterwarnings('ignore')


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
    path = "./shap_output/{}"

    REPO_PATH = config["general"]["repo_path"]
    M_DATE = config["general"]["start_date"]

    AUTOGLUON_OD = False

    dataframe_list = []

    #################################
    # load the datasets and store them within a list
    ################################
    for dataset in datasets:

        # assign dataset
        DATASET = dataset

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

        # extract filename
        FILE_PATH_IN = str(FILE_PATH_IN)
        DELETE_FILE_PATH = FILE_PATH_IN+"/"
        working_file = str(working_file)
        input_filename_with_type = working_file.replace(DELETE_FILE_PATH, "")
        input_filename_without_type = Path(input_filename_with_type).stem
        date_input_filename = "{}-{}".format(M_DATE, input_filename_without_type)
        # print('Input filename: {}'.format(date_input_filename))

        # File path for storing pictures and data
        FILE_PATH_OUT = Path(REPO_PATH, 'measurements', DATASET)
        create_path(path = FILE_PATH_OUT, verbose = False)

        # File path for storing pictures
        FILE_PATH_PICS = Path(REPO_PATH, 'measurements', DATASET, 'pictures')
        create_path(path = FILE_PATH_PICS, verbose = False)

        # File path for storing pictures
        FILE_PATH_PICS = Path(REPO_PATH, 'measurements', DATASET, 'pictures', date_input_filename)
        create_path(path = FILE_PATH_PICS, verbose = False)

        FILE_PATH_PICS = str(FILE_PATH_PICS)

            
        # Delete unnamed / index column
        if set(['Unnamed: 0']).issubset(dataset_df.columns):
            dataset_df = dataset_df.drop('Unnamed: 0', axis=1)

        # add model column
        dataset_df['model'] = model_name

        # add OD for each dataframe
        dataset_df = perform_outlier_detection(dataset_df, 
                                               model_name, 
                                               FILE_PATH_PICS,
                                               config)

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

    print(f"Length of merged dataframe: {len(merged_df)}")
    print(f"Number of features of merged data frame: {merged_df.shape[1]}")

    # Save the DataFrame as a CSV file with the NaN values
    filename = "{}-{}-{}.{}".format("./data/merged-files/merged-files-final", M_DATE, 'NaN','csv')
    merged_df.to_csv(filename, index=False)
    # merged_df.to_csv("./data/merged-files/merged-files-final-2023-10-15-NaN.csv", index=False)
    
    # Delete unnamed / index column
    if set(['Unnamed: 0']).issubset(merged_df.columns):
        data = merged_df.drop('Unnamed: 0', axis=1)
    else:
        data = merged_df.copy()

    ################################
    # Featuretools
    ################################

    # create featuretools EntitySet
    es = ft.EntitySet("merged_data_frame")

    data = data.reset_index()

    es.add_dataframe(dataframe_name="data",
                 index="index",
                 dataframe=data)

    ####
    # Run deep feature synthesis
    ####
    feature_matrix, features = ft.dfs(entityset = es,
                               target_dataframe_name="data")
    
    ###
    # Remove Highly Null Features with a threshold of 0.9
    ###
    new_data = ft.selection.remove_highly_null_features(feature_matrix, pct_null_threshold=0.9)

    ###
    # Remove Single Value Features
    # This is a kind of duplicate detection for features. 
    # Another situation we might run into is one where our calculated features don’t have any variance. 
    # In those cases, we are likely to want to remove the uninteresting features. For that, we use remove_single_value_features.
    ###

    es = ft.EntitySet("new_data_frame")

    es.add_dataframe(dataframe_name="new_data",
                 index="index",
                 dataframe=new_data)
    
    feature_matrix, features = ft.dfs(entityset = es,
                               target_dataframe_name="new_data")
    
    new_fm, new_features = remove_single_value_features(feature_matrix, features=features, count_nan_as_value=True)

    ###################################
    # Remove Highly Correlated Features
    ###################################
    new_fm, new_features = remove_highly_correlated_features(feature_matrix, features=features, pct_corr_threshold=0.9)

    ############################
    # Replace NaN values by 0
    ############################

    # Get the current column names
    column_names = new_fm.columns.tolist()

    # Specify the names/columns with no replacement
    names_to_delete = ['model', 'extension', 'location', 'working_hours', 'const_year', 'price']

    # Create a list of column where the NaN values should be replaced by 0
    columns_to_replace = [col for col in column_names if col not in names_to_delete]
    # print(f"Column names to replace: {columns_to_replace}")

    # Replace NaN values with zeros
    new_fm[columns_to_replace] = new_fm[columns_to_replace].fillna(0)

    ##########################################
    # Drop rows with remaining NaN values
    # TODO: use imputation techniques for 
    # missing values if possible
    ##########################################
    new_fm = new_fm.dropna()

    ################################################
    # Use Autogluon for anomaly / outlier detection
    ################################################

    if AUTOGLUON_OD == True:

        # Split the data into training and test set
        X_train, X_test = train_test_split(new_fm, test_size=0.2, random_state=42)

        # This parameter specifies how many standard deviations above mean anomaly score are considered
        # to be anomalies (only needed for visualization, does not affect scores calculation).
        threshold_stds = 2

        target_col = 'price'

        #####
        # ATTENTION: theres a bug in the detect_anomalies function -> it can be fixed as described here : https://github.com/autogluon/autogluon/issues/3401
        # iloc needs to be changed to loc in the following two files:
        # eda.auto.simple.py line:1667
        # _state.anomaly_detection.anomalies[ds] = df.iloc[anomaly_idx].join(anomaly_scores)
        # _state.anomaly_detection.anomalies[ds] = df.loc[anomaly_idx].join(anomaly_scores)
        # eda.analysis.anomaly.py line:346
        # rows=args[dataset].iloc[dataset_row_ids],
        # rows=args[dataset].loc[dataset_row_ids],
        #####

        state = auto.detect_anomalies(
            train_data= X_train,
            test_data=X_test,
            label=target_col,
            threshold_stds=threshold_stds,
            bps_flag=False, # Don't use bps_flag=True - it's using pre-trained models which aren't loading in newer versions of sklearn -> https://github.com/autogluon/autogluon/pull/3406
            return_state=True,
            show_top_n_anomalies=None,
            explain_top_n_anomalies=None,
            show_help_text=False,
            fig_args=None
        )

        # get the train and test anomalies
        train_anomaly = state.anomaly_detection.anomalies.train_data
        test_anomaly = state.anomaly_detection.anomalies.test_data

        #######################################
        # drop the anomalies from the dataset
        #######################################
        test_indeces = test_anomaly.index.values
        # print(test_indeces)
        data1 = new_fm.drop(test_indeces)
        train_indices = train_anomaly.index.values
        # print(train_indices)
        data = data1.drop(train_indices)

        print(f"Length of preprocessed dataframe: {len(data)}")
        print(f"Number of features of preprocessed data frame: {data.shape[1]}")

    else:
        data = new_fm.copy()

    ##############################
    # Save distribution plots after preprocessing
    ##############################
    # save construction year versus price plot
    data.plot(kind='scatter', x = 'const_year', y = 'price')
    # Set labels for x and y axes
    plt.xlabel('Construction year')
    plt.ylabel('Price')
    plt.title('Price versus construction year')
    plt.savefig(path.format('cons_year_price.pdf'),dpi=100,bbox_inches='tight')

    # save working hours versus price plot
    data.plot(kind='scatter', x = 'working_hours', y = 'price')
    # Set labels for x and y axes
    plt.xlabel('Working hours')
    plt.ylabel('Price')
    plt.title('Price versus working hours')
    plt.savefig(path.format('working_houers_price.pdf'),dpi=100,bbox_inches='tight')

    #########################################
    # Calculate feature importance via SHAP
    ########################################

    y = data['price']
    # print(y.head())

    X_cat = data.drop('price', axis=1)

    #fit encoder
    enc = OneHotEncoder()
    enc.fit(X_cat)

    #transform categorical features
    X_encoded = enc.transform(X_cat).toarray()

    #create feature matrix
    feature_names = X_cat.columns
    new_feature_names = enc.get_feature_names_out(feature_names)

    X = pd.DataFrame(X_encoded, columns= new_feature_names)

    # define and train the model
    model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
    model.fit(X, y)

    #Get predictions
    y_pred = model.predict(X)

    #Get predictions
    # print(mean_absolute_percentage_error(y, y_pred))

    ######################
    # Standard SHAP values
    ######################
    # get shap values
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[1], show=False)
    plt.savefig(path.format('standard_shap_py.png'),dpi=100,bbox_inches='tight')

    ######################
    # SHAP for catgeorical features 
    ######################
    #get number of unique categories for each feature 
    n_categories = []
    for feat in feature_names[:-1]:
        n = X_cat[feat].nunique()
        n_categories.append(n)

    new_shap_values = []
    for values in shap_values.values:
        
        #split shap values into a list for each feature
        values_split = np.split(values , np.cumsum(n_categories))
        
        #sum values within each list
        values_sum = [sum(l) for l in values_split]
        
        new_shap_values.append(values_sum)

    #replace shap values
    shap_values.values = np.array(new_shap_values)

    #replace data with categorical feature values 
    new_data = np.array(X_cat)
    shap_values.data = np.array(new_data)
    # print(shap_values)

    #update feature names
    shap_values.feature_names = list(X_cat.columns)

    # get the SHAP values
    vals = np.abs(shap_values.values).mean(0)

    # calculate the percentage values
    perc_vals = vals / sum(vals)

    feature_importance = pd.DataFrame(list(zip(feature_names, vals, perc_vals)), columns=['feature_name','feature_importance_vals','feature_importance_percentage'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    filename = "{}{}{}{}.{}".format('./measurements/summary/', M_DATE, '/', 'feature-importance', 'csv')
    feature_importance.to_csv(filename, index=False)

    # Sort the DataFrame by the 'Values' column in descending order for correct plot
    feature_importance.sort_values('feature_importance_percentage', inplace=True, ascending=True)

    # Plotting
    plt.figure()
    plt.barh(feature_importance['feature_name'], feature_importance['feature_importance_percentage'],  color='r')
    # feature_importance.plot(kind='barh',y='faeture_name', x='feature_importance_percentage', color='r')

    # Adding labels and title
    plt.xlabel('Feature importance in %')
    plt.ylabel('Features')
    plt.title('Feature importance')

    plt.savefig(path.format('feature_importance_py.png'),dpi=100,bbox_inches='tight')

    # waterfall plot
    plt.figure()
    shap.plots.waterfall(shap_values[1], show=False)

    plt.savefig(path.format('category_shap_py.png'),dpi=100,bbox_inches='tight')

    #Mean SHAP
    plt.figure()
    shap.plots.bar(shap_values,show=False)

    plt.savefig(path.format('mean_shap_py.png'),dpi=100,bbox_inches='tight')

    # Beeswarm
    plt.figure()
    shap.plots.beeswarm(shap_values ,show=False)

    plt.savefig(path.format('beeswarm_py.png'),dpi=100,bbox_inches='tight')

    #get shaply values and data
    model_values = shap_values[:,"model"].values
    model_data = shap_values[:,"model"].data


    #split odor shap values based on odor category
    model_categories = list(set(model_data))

    model_groups = []
    for o in model_categories:
        relevant_values = model_values[model_data == o]
        model_groups.append(relevant_values)
        
    #replace categories with labels
    model_labels = {'1':'308',
                '2':'320', 
                '3':'323', 
                '4':'329', 
                '5':'330', 
                '6':'336', 
                '7':'950', 
                '8':'966',
                '9':'D6', 
                '10':'M318'}

    labels = [model_labels[u] for u in model_labels]

    #plot boxplot
    plt.figure(figsize=(8, 5))

    plt.boxplot(model_groups,labels=labels)

    plt.ylabel('Shap values',size=15)
    plt.xlabel('Odor',size=15)

    plt.savefig(path.format('boxplot_py.png'),dpi=100,bbox_inches='tight')

    # Save the DataFrame as a CSV file
    filename = "{}-{}.{}".format("./data/merged-files/merged-files-final", M_DATE,'csv')
    data.to_csv(filename, index=False)

    ########################
    # build final dataset
    ########################
    # extract/keep features with more than 1%
    selected_values = feature_importance.loc[feature_importance['feature_importance_percentage'] > 0.01, 'feature_name']

    selected_values = selected_values.append(pd.Series(['price']))

    # create final dataframe with the selected features
    selectetd_data = data[selected_values].copy()

    time.sleep(5)

    # Save the DataFrame as a CSV file
    filename = "{}-{}.{}".format("./data/merged-files/merged-files-final-selected-features", M_DATE,'csv')
    selectetd_data.to_csv(filename, index=True)

    return(selectetd_data)
