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

import re

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

    # print(merged_df)

    return(merged_df)
