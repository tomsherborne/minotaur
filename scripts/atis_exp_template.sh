#!/usr/bin/env bash
#SBATCH -o /home/%u/slogs/%A_%a.out
#SBATCH -e /home/%u/slogs/%A_%a.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH -t 5-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=4  # number of cpus to use - there are 32 on each node.

# Activate Conda
source /home/${USER}/miniconda3/bin/activate cuda112

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y_%H:%M');
echo ${dt}

# Env variables
export MKL_THREADING_LAYER=GNU
export STUDENT_ID=${USER}
export EXP_ROOT="${HOME}/sp/struct"

# Run the python script that will train our network
export LINE="`sed \"${SLURM_ARRAY_TASK_ID}q;d\" $1`"
export EXP_CONFIG="$(cut -d ";" -f1 <<< "${LINE}")"
export EXP_NAME="$(basename "${EXP_CONFIG}" .json)"

echo "BUILDING CONFIG========"
echo "Experiment ${EXP_NAME} with config ${EXP_CONFIG} running"
export SERIAL_DIR="${HOME}/exps/${EXP_NAME}"

# Serialisation Directory for AllenNLP is the experiment folder
if [[ ! -f ${EXP_CONFIG} ]]; then
    echo "Config file ${EXP_CONFIG} is invalid, quitting..."
    exit 1
fi

# Ensure the scratch home exists and CD to the experiment root level.
mkdir -p "${SERIAL_DIR}"
cd "${EXP_ROOT}" # helps AllenNLP behave

echo "TRAIN========"
allennlp train "${EXP_CONFIG}" \
    --serialization-dir "${SERIAL_DIR}" \
    --include-package codebase \
    --file-friendly-logging \
    --force

rsync -auzhP "${SERIAL_DIR}" "${EXP_ROOT}/runs/cluster/" # Copy output onto headnode

echo "============"
echo "training finished successfully"

echo "TEST========"
TEST_LOCALES="en fr pt es de zh"

for TEST_LOCALE in ${TEST_LOCALES};
do
    echo "Testing for locale ${TEST_LOCALE}"
    export JSON_PRED_INPUT="${EXP_ROOT}/data/multiatis2sql/${TEST_LOCALE}/${TEST_LOCALE}.test.nl"
    export JSON_PRED_OUTPUT="${SERIAL_DIR}/${TEST_LOCALE}.sql.pred.json"
    export TOK_PRED_OUTPUT="${SERIAL_DIR}/${TEST_LOCALE}.sql.pred"

    allennlp predict "${SERIAL_DIR}/model.tar.gz" \
                     "${JSON_PRED_INPUT}" \
                     --cuda-device 0 \
                     --output-file "${JSON_PRED_OUTPUT}" \
                     --include-package codebase \
                     --predictor "seq2seq_anyhead"
done

echo "============"
echo "job finished"
