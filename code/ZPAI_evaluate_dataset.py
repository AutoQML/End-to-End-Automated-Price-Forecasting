# Import necessary packages.
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

def calculate_stats(df: pd.DataFrame, 
                    dataset: str, 
                    filename: str, 
                    creation_date: str, 
                    yaml_file: str) -> None:
    """
    Calculates mean, standard deviation, min./max. value and quartiles for the given dataset and 
    appends the result to the yaml file

    Detailed description

    Parameters
    ----------
    df : pd.DataFrame
        Description of arg1
    machine_model : str
        Description of arg2
    filename : str
        Description
    creation_date : str
        Description
    yaml_file : str
        Description
    
    Returns
    -------


    """

    # open yaml file to check if entry already exists
    with open(yaml_file, 'r') as file: # open the file in append mode
        summary_result_values = yaml.safe_load(file)

    # insert statistics if it does not already exists
    if dataset not in summary_result_values:

        # print('-- Instert statistics to model: ', dataset)
         
        file_mean = int(df['price'].describe()['mean'])
        file_std = int(df['price'].describe()['std'])
        file_min = int(df['price'].describe()['min'])
        file_max = int(df['price'].describe()['max'])
        file_25 = int(df['price'].describe()['25%'])
        file_50 = int(df['price'].describe()['50%'])
        file_75 = int(df['price'].describe()['75%'])
        iqr = int(file_75 - file_25) #Interquartile range
        if((file_25 - (1.5*iqr)) < 0):
            fence_low = 0
        else:
            fence_low = int(file_25 - (1.5*iqr))
        fence_high = int(file_75 + (1.5*iqr))

        dict_file_stats = {dataset :{'input_file_name': filename, 
                                    'input_file_size': len(df), 
                                    'input_file_creation_date': creation_date, 
                                    'input_file_mean' : file_mean,
                                    'input_file_std': file_std, 
                                    'input_file_min' : file_min,
                                    'input_file_max': file_max, 
                                    'input_file_25' : file_25, 
                                    'input_file_50': file_50, 
                                    'input_file_75' : file_75,
                                    'input_file_iqr' : iqr,
                                    'input_file_fl' : fence_low,
                                    'input_file_fh' : fence_high}}

        with open(yaml_file, 'a') as file: # open the file in append mode
            yaml.dump(dict_file_stats, file, default_flow_style=False)

    # else:
    #     print('Statistics to model: {} already exists!'.format(dataset))

def evaluate_data(dataset: str,
                            measurement: int,
                            GLOBAL_TXT_SUMMARY_FILE: str, 
                            GLOBAL_YAML_SUMMARY_FILE: str,
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


    # assign dataset
    DATASET = dataset
    NUM_OF_MEASUREMENT = measurement
    GLOBAL_TXT_SUMMARY_FILE = GLOBAL_TXT_SUMMARY_FILE
    GLOBAL_YAML_SUMMARY_FILE = GLOBAL_YAML_SUMMARY_FILE
    

    # Get parameters from configuration file
    M_DATE = config["general"]["start_date"]
    REPO_PATH = config["general"]["repo_path"]
    EVAL_DATASET_VARIANCE = config["general"]["evaluate_dataset_variance"]

    if EVAL_DATASET_VARIANCE == True:
        RANDOM_STATE = measurement
    else:
        RANDOM_STATE = config["general"]["random_state"]


    # set path to CSV data files 
    FILE_PATH_IN = Path(REPO_PATH, 'data', DATASET)

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

    # write name of input file to global summary.txt
    with open(GLOBAL_TXT_SUMMARY_FILE, "a") as f:
        f.write("Input file for " + 
                DATASET + ": " + input_filename_with_type + 
                " with size: " + str(len(dataset_df)) + 
                " created: " + str(file_creation_date) + "\n")
        
    # Delete unnamed / index column
    if set(['Unnamed: 0']).issubset(dataset_df.columns):
        dataset_df = dataset_df.drop('Unnamed: 0', axis=1)

    ######################################
    # CREATE PATHS
    ######################################

    # File path for storing pictures and data
    FILE_PATH_OUT = Path(REPO_PATH, 'measurements', DATASET)
    create_path(path = FILE_PATH_OUT, verbose = False)

    # File path for storing pictures
    FILE_PATH_PICS = Path(REPO_PATH, 'measurements', DATASET, 'pictures')
    create_path(path = FILE_PATH_PICS, verbose = False)
    
    # File path for storing data
    FILE_PATH_DATA = Path(REPO_PATH, 'measurements', DATASET, 'data')
    create_path(path = FILE_PATH_DATA, verbose = False)

    # File path for storing pictures
    FILE_PATH_PICS = Path(REPO_PATH, 'measurements', DATASET, 'pictures', date_input_filename)
    create_path(path = FILE_PATH_PICS, verbose = False)

    # File path for storing data
    FILE_PATH_DATA = Path(REPO_PATH, 'measurements', DATASET, 'data', date_input_filename)
    create_path(path = FILE_PATH_DATA, verbose = False)

    # File path for storing pictures
    FILE_PATH_PICS = Path(REPO_PATH, 'measurements', DATASET, 'pictures', date_input_filename, str(NUM_OF_MEASUREMENT))
    create_path(path = FILE_PATH_PICS, verbose = False)

    # File path for storing data
    FILE_PATH_DATA = Path(REPO_PATH, 'measurements', DATASET, 'data', date_input_filename, str(NUM_OF_MEASUREMENT))
    create_path(path = FILE_PATH_DATA, verbose = False)

    # create summary txt file
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'summary','txt')
    SUMMARY_FILE = Path(FILE_PATH_DATA, filename)
    if not Path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "w") as f:
            f.write("Summary for: \n" + input_filename_without_type + "\n")
        # print("Summary file " , SUMMARY_FILE ,  " Created ")
    else:
        with open(SUMMARY_FILE, "a") as f:
            f.write("Summary for: \n" + input_filename_without_type + "\n")
        # print("Summary file " , SUMMARY_FILE ,  " already exists")

    # store dataset_df.head() as csv
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'head','csv')
    OVERVIEW_CSV = Path(FILE_PATH_DATA, filename)
    dataset_df.head(20).to_csv(OVERVIEW_CSV)

    ######################################
    # CREATE & SAVE PLOTS
    ######################################

    FILE_PATH_PICS = str(FILE_PATH_PICS)

    # Histogram
    dataset_df.hist(bins=40, figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'histogram','png')
    title = str("Histogram for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    # Construction year vs. Price scatter plot
    dataset_df.plot(kind='scatter', x='const_year', y='price', figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'year-price','png')
    title = str("Year / price for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    # Working hours vs. Price scatter plot
    dataset_df.plot(kind='scatter', x='working_hours', y='price', figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'hours-price','png')
    title = str("Hours / price for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    ###################################
    # CALCULATE AND SAVE STATS OF INPUT FILE
    ###################################

    calculate_stats(df = dataset_df, 
                    dataset = DATASET,
                    filename = input_filename_with_type,  
                    creation_date = file_creation_date,
                    yaml_file = GLOBAL_YAML_SUMMARY_FILE)

    ######################################
    # CREATE & SAVE PLOTS OF "CLEAN" INPUT FILE
    ######################################

    # Plot the date after sanity check and cleanup
    # store working hours / price scatter plot
    dataset_df.plot(kind='scatter', x='working_hours', y='price', figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'hours-price-after-clean','png')
    title = str("Hours / price after clean for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    # store construction year / price scatter plot
    dataset_df.plot(kind='scatter', x='const_year', y='price', figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'year-price-after-clean','png')
    title = str("Year / price after clean for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    # store construction year / working hours scatter plot
    dataset_df.plot(kind='scatter', x='const_year', y='working_hours', figsize=(10,7))
    filename = "{}-{}-{}.{}".format(M_DATE,input_filename_without_type,'year-hours-after-clean','png')
    title = str("Year / hours after clean for {}".format(input_filename_without_type))
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)
    plt.close()

    ######################################
    # NEXT STEP
    ######################################

    # call next function in the ML pipeline: prepare_data_for_ml()

    prepare_data_for_ml(df_dataset = dataset_df,
                        dataset_name = DATASET, 
                        file_path_pics = FILE_PATH_PICS, 
                        file_path_data = FILE_PATH_DATA, 
                        input_filename = input_filename_without_type, 
                        summary_file = SUMMARY_FILE, 
                        config = config,
                        random_state = RANDOM_STATE,
                        measurements = NUM_OF_MEASUREMENT)

