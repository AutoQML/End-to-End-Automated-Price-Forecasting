!/bin/bash

clear

# base_env = conda info | grep -i 'base environment'

# get current date
START_DATE=`date +%F`
echo $START_DATE

# Get the directory where the script is located
script_directory=$(dirname "$0")
echo "Script directory: $script_directory"

# get conda base directory path
CONDA_BASE=$(conda info --base)
echo "Base env: $CONDA_BASE"
# source /Users/StuehlerH/opt/anaconda3/etc/profile.d/conda.sh
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "Starting control script"
echo "Operating system: $OSTYPE"

# print current conda env
echo "Conda env at start: $CONDA_DEFAULT_ENV"

# deactivate current conda env
conda deactivate
echo "Conda env after deactivate: $CONDA_DEFAULT_ENV"

# activate automl-autosklearn conda env
conda activate automl-autosklearn
echo "Conda env after activate automl-autosklearn: $CONDA_DEFAULT_ENV"
echo "Conda env: $CONDA_DEFAULT_ENV"

# set variables
MEASUREMENTS=1
DATASET='Caterpillar-320'
PCA_NUM=0

echo "Num of measurements: $MEASUREMENTS"
echo "Dataset: $DATASET"


# start computation if os == darwin (macos)
if [[ $OSTYPE == 'darwin'* ]]
then
    echo 'macOS'
    if [ $CONDA_DEFAULT_ENV == "automl-autosklearn" ]
    then
        python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms manual nn --datasets $DATASET --pca $PCA_NUM --measurements $MEASUREMENTS --document_results True
    fi
fi

# start computation if os == linux 
if [[ $OSTYPE == 'linux'* ]] 
then
    echo 'linux'
    if [ $CONDA_DEFAULT_ENV == "automl-autosklearn" ]
    then
                python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms manual nn autosklearn flaml --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM --autosk_time_for_task 600 --autosk_runtime_limit 60 --document_results False
    fi

    conda deactivate
    echo "Conda env after deactivate: $CONDA_DEFAULT_ENV"

    conda activate autogluon
    echo "Conda env after activate autogluon: $CONDA_DEFAULT_ENV"

    if [ $CONDA_DEFAULT_ENV == "autogluon" ]
    then
        python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms autogluon --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM --document_results False
    fi

    conda deactivate

    conda activate autokeras-2
    echo "Conda env after activate autokeras-2: $CONDA_DEFAULT_ENV"

    if [ $CONDA_DEFAULT_ENV == "autokeras-2" ]
    then
        python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms autokeras --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM --document_results False
    fi

    conda deactivate
fi
