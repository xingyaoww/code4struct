#!/bin/bash
set -x

DATASET_PATH=data/extraction/ace
DATASET_SUBSET=ACE05-E

function infer() {
    # arguments: version, n, temperature, k_shots
    DATASET_SUBSET=$1 # e.g., ACE05-E
    MODEL=$2 # e.g., code-davinci-002, text-davinci-002
    VERSION=$3 # e.g., v0-baseline, v1-reduce-hallucination
    N=$4
    TEMPERATURE=$5
    K_SHOTS=$6 # default to 0
    EXTRA_NAME=$7 # default to ""
    EXTRA_FLAGS=$8 # default to ""

    # if k_shots > 0
    if [ $K_SHOTS -gt 0 ]; then
        OUTPUT_DIRNAME="${VERSION}-${K_SHOTS}shot-n${N}-t${TEMPERATURE}"
    else
        OUTPUT_DIRNAME="${VERSION}-n${N}-t${TEMPERATURE}"
    fi

    PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/model/openai-model.py \
        --input-filepath ${DATASET_PATH}/processed/${DATASET_SUBSET}/${VERSION}/test.jsonl \
        --input-train-filepath ${DATASET_PATH}/processed/${DATASET_SUBSET}/${VERSION}/train.jsonl \
        --output-dir ${DATASET_PATH}/inferred/${DATASET_SUBSET}/codex/${OUTPUT_DIRNAME}${EXTRA_NAME}/ \
        --model $MODEL \
        --n-generations $N --temperature $TEMPERATURE --k-shot $K_SHOTS \
        --batch-size 20 \
        ${EXTRA_FLAGS}
}

# v6.4-baseline+trigger+hierarchy
infer ACE05-E code-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 0
infer ACE05-E code-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 1
infer ACE05-E code-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 10
infer ACE05-E code-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 20
infer ACE05-E code-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 50

# Cross-sibling Transfer hierarchical examples
infer ACE05-E code-davinci-002 v9.2-baseline+trigger+1hierarchyexample+siblingonly 1 0.0 1 
infer ACE05-E code-davinci-002 v9.2-baseline+trigger+10hierarchyexample+siblingonly 1 0.0 10

# Compare with text-davinci-002
infer ACE05-E text-davinci-002 v5.1-baseline+puretext+trigger 1 0.0 0
infer ACE05-E text-davinci-002 v5.1-baseline+puretext+trigger 1 0.0 1
infer ACE05-E text-davinci-002 v5.1-baseline+puretext+trigger 1 0.0 5
infer ACE05-E text-davinci-002 v5.1-baseline+puretext+trigger 1 0.0 10
infer ACE05-E text-davinci-002 v5.1-baseline+puretext+trigger 1 0.0 20

infer ACE05-E text-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 0
infer ACE05-E text-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 1
infer ACE05-E text-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 10
infer ACE05-E text-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 20
infer ACE05-E text-davinci-002 v6.4-baseline+trigger+hierarchy 1 0.0 50
