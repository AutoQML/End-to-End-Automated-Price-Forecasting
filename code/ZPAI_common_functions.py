import pandas as pd
from datetime import date
from pathlib import Path
import yaml
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns

# load CSV data file
def load_csv_data(csv_path):
    return pd.read_csv(csv_path, delimiter=',')

# Define the helper function `display_scores` to print the results of the cross validation
def display_scores(scores, f):
        # f.write('\n\n')
        # f.write('Scores: ' + str(scores))
        f.write('\n')
        f.write('Mean: \t' + str(int(scores.mean())))
        f.write('\n')
        f.write('StD: \t' + str(int(scores.std())))
        f.write('\n\n')

def read_yaml(path):
    """
    Safe load yaml file
    """
    with open(path, 'r') as f:
        file = yaml.safe_load(f)
    return file

def get_current_date() -> str:
    """
    Get current date as string
    """
    today = date.today()
    m_date = today.strftime("%Y-%m-%d")
    return m_date

def create_path(path: str, verbose: bool):
    """
    Checks if directory exists and, if not, creates directory
    """
    if not Path.exists(path):
        Path.mkdir(path)
        if verbose:
            print("Directory " , path ,  " Created ")
    else:
        if verbose:    
            print("Directory " , path ,  " already exists")

def calculate_scores(y_test, y_predict):
    # Calculate MAE and MEPE according to https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-percentage-error
    mean_abs_error = mean_absolute_error(y_test, y_predict)
    mean_abs_percentage_error = mean_absolute_percentage_error(y_test, y_predict)
    r2_score_value = r2_score(y_test, y_predict) 

    # calculate RMSE value and its derivative according to https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalization
    # RMSE
    root_mean_squared_error = np.sqrt(mean_squared_error(y_test, y_predict))

    return(mean_abs_error, mean_abs_percentage_error, r2_score_value, root_mean_squared_error)


###########################
# Perform outlier detection
###########################
def perform_outlier_detection(dataset_df: pd.DataFrame,
                              model_name: str,
                              file_path_pics: str,
                              config: dict):
    
    M_DATE = config["general"]["start_date"]
    FILE_PATH_PICS = file_path_pics

    # boxplot construction year
    plt.figure(figsize=(15, 20))  # Optional: set the size of the figure
    filename = "{}-{}-{}.{}".format(M_DATE, model_name,'construction-year-boxplot-sns', 'pdf')
    sns.boxplot(y='const_year', data=dataset_df)
    title = "{}-{} {}".format('Caterpillar', model_name, 'construction year boxplot')
    plt.ylabel('Construction year')
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # boxplot price
    plt.figure(figsize=(15, 20))  # Optional: set the size of the figure
    filename = "{}-{}-{}.{}".format(M_DATE, model_name,'price-boxplot-sns', 'pdf')
    sns.boxplot(y='price', data=dataset_df)
    title = "{}-{} {}".format('Caterpillar', model_name, 'price boxplot')
    plt.ylabel('Price')
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # boxplot working hours
    plt.figure(figsize=(15, 20))  # Optional: set the size of the figure
    filename = "{}-{}-{}.{}".format(M_DATE, model_name,'working-hours-boxplot-sns', 'pdf')
    sns.boxplot(y='working_hours', data=dataset_df)
    title = "{}-{} {}".format('Caterpillar', model_name, 'working hours boxplot')
    plt.ylabel('Working hours')
    plt.title(title)
    plt.savefig(FILE_PATH_PICS+'/'+filename)

    # calculate IQR for price feature
    q1 = pd.DataFrame(dataset_df['price']).quantile(0.25)[0]
    q3 = pd.DataFrame(dataset_df['price']).quantile(0.75)[0]
    iqr_price = q3 - q1 #Interquartile range
    fence_low_price = q1 - (1.5*iqr_price)
    fence_high_price = q3 + (1.5*iqr_price)
    # print(q1, q3, iqr_price, fence_low_price, fence_high_price)

    # calculate IQR for working hours feature
    q1 = pd.DataFrame(dataset_df['working_hours']).quantile(0.25)[0]
    q3 = pd.DataFrame(dataset_df['working_hours']).quantile(0.75)[0]
    iqr_wh = q3 - q1 #Interquartile range
    fence_low_wh = q1 - (1.5*iqr_wh)
    fence_high_wh = q3 + (1.5*iqr_wh)
    # print(q1, q3, iqr_wh, fence_low_wh, fence_high_wh)

    # Delete outliers
    # Define the conditions for price deletion
    condition = (dataset_df['price'] > fence_high_price)
    # Delete rows based on the combined condition
    dataset_df = dataset_df[~condition]

    # Define the conditions for working hours deletion
    condition = (dataset_df['working_hours'] > fence_high_wh)
    # Delete rows based on the combined condition
    dataset_df = dataset_df[~condition]

    return dataset_df