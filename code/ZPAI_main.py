import argparse
import yaml
import platform
from pathlib import Path
import os

from ZPAI_common_functions import get_current_date, create_path, read_yaml
from ZPAI_evaluate_dataset import evaluate_data
from ZPAI_document_results_docx import document_results_docx

def create_parser():
    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument("--pca", type = int, help = "Number of selected PCA components")
    parser.add_argument("--algorithms", choices = ["manual", "nn", "autosklearn", "autogluon", "flaml"], nargs="*", help = "Type of algorithms to run.")
    parser.add_argument("--datasets", nargs='+', help = "Enter dataset(s) - use the subdirectory names of the data folder", required=True)
    parser.add_argument("--outlier_detection", choices = ["True", "False"], help = "Remove outliers from dataset")
    parser.add_argument("--document_results", choices = ["True", "False"], help = "Document the results")
    parser.add_argument("--autosk_time_for_task", type = int, help = "Time limit in seconds for the search of appropriate models")
    parser.add_argument("--autosk_runtime_limit", type = int, help = "Time limit for a single call to the machine learning model")
    parser.add_argument("--start_date", type = str, help = "Start date of measurement")
    parser.add_argument("--measurements", type = int, help = "Number of measurements")
    parser.add_argument("--automl_preprocessing", choices = ["True", "False"], help = "Set autoML preprocessing")
    parser.add_argument("--evaluate_dataset_variance", choices = ["True", "False"], help = "Set evaluation of data set variance")
    parser.add_argument("--random_state", type = int, help = "Random state")

    return parser

def get_config_from_parser(parser, config):
    args = parser.parse_args()

    # List of datasets
    if args.datasets:
        config["general"]['datasets'] = args.datasets

    # Algorithms
    if args.algorithms:
        config["general"]["algorithms"] = args.algorithms

    # Outlier Detection
    if args.outlier_detection:
        if args.outlier_detection == "True":
            config["general"]["bin_outlier_detect"] = True
        else:
            config["general"]["bin_outlier_detect"] = False

    # Documentation
    if args.document_results:
        if args.document_results == "True":
            config["general"]["documentation"] = True
        else:
            config["general"]["documentation"] = False

    # PCA Num
    if args.pca:
        config["general"]["pca_num"] = args.pca

    # Autosklearn
    if args.autosk_time_for_task:
        config["autosklearn"]["params"]["time_for_task"] = args.autosk_time_for_task

    if args.autosk_runtime_limit:
        config["autosklearn"]["params"]["run_time_limit"] = args.autosk_runtime_limit

    # date
    if args.start_date:
        config["general"]["start_date"] = args.start_date

    # Number of measuremets
    if args.measurements:
        config["general"]["measurement_num"] = args.measurements

    # autoML preprocessing
    if args.automl_preprocessing:
        if args.automl_preprocessing == 'True':
            config["general"]["automl_preprocessing"] = True
        else:
            config["general"]["automl_preprocessing"] = False

    # Set evaluation of data set variance
    if args.evaluate_dataset_variance:
        if args.evaluate_dataset_variance == 'True':
            config["general"]["evaluate_dataset_variance"] = True
        else:
            config["general"]["evaluate_dataset_variance"] = False

    # Random State
    if args.random_state:
        config["general"]["random_state"] = args.random_state

    return config


def main():
    parser = create_parser()

    ######################################
    # LOAD CONFIGURATIONS
    ######################################
    REPO_PATH = Path(__file__).parents[1]

    # Load configuration files
    general_conf = read_yaml(REPO_PATH / 'conf/general_config.yml')
    feature_conf = read_yaml(REPO_PATH / 'conf/feature_config.yml')
    autosklearn_conf = read_yaml(REPO_PATH / 'conf/auto_sklearn_config.yml')

    # Create global configuration file
    CFG = dict()
    CFG["general"] = general_conf
    CFG["features"] = feature_conf
    CFG["autosklearn"] = autosklearn_conf

    CFG["general"]["repo_path"] = Path(__file__).parents[1]
    CFG["general"]["start_date"] = get_current_date()
    CFG["general"]["operating_system"] = platform.system()
    # add python environment
    CFG["general"]["python_env"] = os.environ['CONDA_DEFAULT_ENV']

    CFG = get_config_from_parser(parser, CFG)

    # Get general configuration parameters
    REPO_PATH = CFG["general"]["repo_path"]
    PCA_NUM = CFG["general"]['pca_num']   # get the PCA number
    RANDOM_STATE = CFG["general"]['random_state'] # get the random state
    BIN_OUTLIER = CFG["general"]['bin_outlier_detect'] # get bin outlier detection & deletion state
    DATASETS = CFG["general"]['datasets'] # get list of datasets
    M_DATE = CFG["general"]["start_date"]
    DOCUMENTATION = CFG["general"]["documentation"]
    ALGORITHMS = CFG["general"]["algorithms"]
    NUM_OF_MEASUREMENTS = CFG["general"]["measurement_num"]

    ######################################
    # INITIALIZE COMMON SUMMARY
    ######################################

    # File path for measurements
    MEASUREMENT_FILE_PATH = Path(REPO_PATH, 'measurements')
    if not Path.exists(MEASUREMENT_FILE_PATH):
        create_path(path = MEASUREMENT_FILE_PATH, verbose = False)

    # File path for the common summary
    SUMMARY_FILE_PATH = Path(REPO_PATH, 'measurements', 'summary')
    if not Path.exists(SUMMARY_FILE_PATH):
        create_path(path = SUMMARY_FILE_PATH, verbose = False)

    # File path within the summary directory for each measurement
    EXPLICIT_SUMMARY_FILE_PATH = Path(REPO_PATH, 'measurements', 'summary', M_DATE)
    if not Path.exists(EXPLICIT_SUMMARY_FILE_PATH):
        create_path(path = EXPLICIT_SUMMARY_FILE_PATH, verbose = False)

    # create summary txt file
    filename = "{}-{}.{}".format(M_DATE,'summary','txt')
    GLOBAL_TXT_SUMMARY_FILE = Path(EXPLICIT_SUMMARY_FILE_PATH, filename)
    if not Path.exists(GLOBAL_TXT_SUMMARY_FILE):
        with open(GLOBAL_TXT_SUMMARY_FILE, "w") as f:
            f.write("Measuremt date: " + M_DATE + "\n")
            f.write("Random seed: " + str(RANDOM_STATE) + "\n")
            f.write("PCA number: " + str(PCA_NUM) + "\n")
            f.write("Datasets: " + str(DATASETS) + "\n")
            if "autosklearn" in ALGORITHMS:
                f.write("Auto-sklearn runtime " + str(autosklearn_conf['params']['time_for_task']) + "\n")
                f.write("Auto-sklearn limit : " + str(autosklearn_conf['params']['run_time_limit']) + "\n")


    # create summary yaml file
    filename = "{}-{}.{}".format(M_DATE,'summary','yml')
    GLOBAL_YAML_SUMMARY_FILE = Path(EXPLICIT_SUMMARY_FILE_PATH, filename)

    if "autosklearn" in ALGORITHMS:
        dict_file = {'measurement_date': M_DATE,
                    'random_seed': RANDOM_STATE,
                    'pca_numbers': PCA_NUM,
                    'number_of_measurements': NUM_OF_MEASUREMENTS,
                    'bin_outlier_detect': BIN_OUTLIER,
                    'autosklearn_runtime': autosklearn_conf['params']['time_for_task'],
                    'autosklearn_limit': autosklearn_conf['params']['run_time_limit'] }
    else:
        dict_file = {'measurement_date': M_DATE,
                    'random_seed': RANDOM_STATE,
                    'pca_numbers': PCA_NUM,
                    'number_of_measurements': NUM_OF_MEASUREMENTS,
                    'bin_outlier_detect': BIN_OUTLIER }

    # create first entry once at the creation of the file
    if not Path.exists(GLOBAL_YAML_SUMMARY_FILE):
        with open(GLOBAL_YAML_SUMMARY_FILE, 'w') as file:
            documents = yaml.dump(dict_file, file)


    ######################################
    # RUN PIPELINE AND ALGORITHMS FOR ALL SELECTED MODELS
    ######################################
    # outmost loop -> configure number of repetitive runs
    for measurement in range(NUM_OF_MEASUREMENTS):

        # print number of measurements
        print('\n Measurement {} of {} with random state {}'.format(measurement+1, NUM_OF_MEASUREMENTS, measurement+1))

        # iterate through all construction machine models
        for count, dataset in enumerate(DATASETS):

            # get model configuration
            print('\n Construction machine model {} of {} - {}'.format(count+1, len(DATASETS), dataset))

            evaluate_data(dataset = dataset,
                                    measurement = measurement + 1,
                                    GLOBAL_TXT_SUMMARY_FILE = GLOBAL_TXT_SUMMARY_FILE,
                                    GLOBAL_YAML_SUMMARY_FILE = GLOBAL_YAML_SUMMARY_FILE,
                                    config = CFG)


    ######################################
    # CREATE WORD FILE WITH ALL RESULTS
    ######################################

    # document results as docx
    if DOCUMENTATION == True:
        document_results_docx(datasets = DATASETS,
                              NUM_OF_MEASUREMENTS = NUM_OF_MEASUREMENTS,
                              GLOBAL_YAML_SUMMARY_FILE = GLOBAL_YAML_SUMMARY_FILE,
                              EXPLICIT_SUMMARY_FILE_PATH = EXPLICIT_SUMMARY_FILE_PATH,
                              config = CFG)

if __name__ == '__main__':
    main()
