from pathlib import Path
from datetime import date
from ZPAI_common_functions import read_yaml

from ZPAI_document_results_docx import document_results_docx

###############################
# Create dokumentation
###############################
# datasets = ['Caterpillar-308', 'Caterpillar-320', 'Caterpillar-323', 'Caterpillar-329', 'Caterpillar-330', 'Caterpillar-336', 'Caterpillar-950', 'Caterpillar-966', 'Caterpillar-D6', 'Caterpillar-M318']
# datasets = ['merged-files']
# datasets = ['Caterpillar-950', 'Caterpillar-966']
# datasets = ['Caterpillar-320']

# define methods to be droped from the scatter plots
# drop_methods = []
drop_methods = ['manual', 'nn']


##############################
# Define data set version
##############################
# Set dataset(s)
datasets = ['merged-files']
# Set date
m_date = '2023-12-07'
# Set number of measuremts
NUM_OF_MEASUREMENTS = 5
# Set file description
# file_description = 'final-selected-features'
file_description = 'final'

# create summary yaml file
# File path within the summary directory for each measurement
EXPLICIT_SUMMARY_FILE_PATH = Path('./measurements', 'summary', m_date)

filename = "{}-{}.{}".format(m_date,'summary','yml')
GLOBAL_YAML_SUMMARY_FILE = Path(EXPLICIT_SUMMARY_FILE_PATH, filename)

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

# CFG["general"]["start_date"] = get_current_date()
CFG["general"]["start_date"] = m_date

# document_results_docx(datasets,m_date, GLOBAL_YAML_SUMMARY_FILE, EXPLICIT_SUMMARY_FILE_PATH)
document_results_docx(datasets,
                        file_description,
                        drop_methods,
                        NUM_OF_MEASUREMENTS = NUM_OF_MEASUREMENTS,
                        GLOBAL_YAML_SUMMARY_FILE = GLOBAL_YAML_SUMMARY_FILE, 
                        EXPLICIT_SUMMARY_FILE_PATH = EXPLICIT_SUMMARY_FILE_PATH, 
                        config = CFG)