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

# set variables
MEASUREMENTS=1
DATASET='merged-files'
PCA_NUM=0
DATA_PREPROCESSING='False'
TIME_FOR_TASK=60
RUNTIME_LIMIT=6

RUN_AUTOSKLEARN='True'
RUN_AUTOGLUON='False'
RUN_FLAML='True'
RUN_AUTOKERAS='False'
RUN_MANUAL='False'
RUN_NN='False'

echo "Num of measurements: $MEASUREMENTS"
echo "Dataset: $DATASET"


# start computation if os == darwin (macos)
if [[ $OSTYPE == 'darwin'* ]]
then
    echo 'macOS'
    # activate automl-autosklearn conda env
    conda activate automl-autosklearn
    echo "Conda env after activate automl-autosklearn: $CONDA_DEFAULT_ENV"

    if [ $CONDA_DEFAULT_ENV == "automl-autosklearn" ]
    then
        python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms manual nn --datasets $DATASET --pca $PCA_NUM --measurements $MEASUREMENTS --document_results True
    fi
fi

# start computation if os == linux 
if [[ $OSTYPE == 'linux'* ]] 
then
    echo 'linux'
    
    ### Auto-Sklearn
    if [ $RUN_AUTOSKLEARN == 'True' ]
    then
        conda activate autosklearn
        echo "Conda env after activate autosklearn: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "autosklearn" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms autosklearn --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM --autosk_time_for_task $TIME_FOR_TASK --autosk_runtime_limit $RUNTIME_LIMIT
        fi

        conda deactivate
    fi

    ### AutoGluon
    if [ $RUN_AUTOGLUON == 'True' ]
    then
        conda activate autogluon
        echo "Conda env after activate autogluon: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "autogluon" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms autogluon --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM
        fi

        conda deactivate
    fi

    ### manual
    if [ $RUN_MANUAL == 'True' ]
    then
        conda activate sklearn
        echo "Conda env after activate sklearn: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "sklearn" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms manual --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM
        fi

        conda deactivate
    fi

    ### nn
    if [ $RUN_NN == 'True' ]
    then
        conda activate sklearn
        echo "Conda env after activate sklearn: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "sklearn" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms nn --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM
        fi

        conda deactivate
    fi


    ### FLAML
    if [ $RUN_FLAML == 'True' ]
    then
        conda activate flaml
        echo "Conda env after activate flaml: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "flaml" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms flaml --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM --autosk_time_for_task $TIME_FOR_TASK --autosk_runtime_limit $RUNTIME_LIMIT 
        fi

        conda deactivate
    fi

    ## AutoKeras
    if [ $RUN_AUTOKERAS == 'True' ]
    then
        conda activate autokeras-2
        echo "Conda env after activate autokeras-2: $CONDA_DEFAULT_ENV"

        if [ $CONDA_DEFAULT_ENV == "autokeras-2" ]
        then
            python $script_directory/code/ZPAI_main.py --start_date $START_DATE --algorithms autokeras --datasets $DATASET --measurements $MEASUREMENTS --pca $PCA_NUM
        fi

        conda deactivate
    fi
fi
