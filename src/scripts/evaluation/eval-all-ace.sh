#!/bin/bash

set -e

function eval_ace() {
    INFER_DIRNAME=$1 # e.g., ACE05-E/codex/v7-baseline+trigger+keywords-n10-t0.5
    GENERATED_FILE=data/extraction/ace/inferred/$INFER_DIRNAME/test.jsonl
    DATA_SUBSET=$(echo $INFER_DIRNAME | cut -f1 -d '/') # e.g., ACE05-E
    TEST_FILE=data/extraction/ace/interim/$DATA_SUBSET/split/test.oneie.json

    set -e

    PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/evaluation/scorer.py \
        --gen-file ${GENERATED_FILE} \
        --test-file $TEST_FILE \
        --ontology-path data/ontology/ace/processed/ace_roles.json \
        --head-only

    pandoc \
        --standalone ${GENERATED_FILE}.head_only.eval.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o ${GENERATED_FILE}.head_only.eval.html \
        --metadata title="$INFER_DIRNAME Evaluation (Head Only)"

    PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/evaluation/scorer.py \
        --gen-file ${GENERATED_FILE} \
        --test-file $TEST_FILE \
        --ontology-path data/ontology/ace/processed/ace_roles.json \
        --head-only --coref

    pandoc \
        --standalone ${GENERATED_FILE}.coref.eval.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o ${GENERATED_FILE}.coref.eval.html \
        --metadata title="$INFER_DIRNAME Evaluation (Head Only & Coref)"

    PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/evaluation/scorer.py \
        --gen-file ${GENERATED_FILE} \
        --test-file $TEST_FILE \
        --ontology-path data/ontology/ace/processed/ace_roles.json

    pandoc \
        --standalone ${GENERATED_FILE}.eval.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o ${GENERATED_FILE}.eval.html \
        --metadata title="$INFER_DIRNAME Evaluation"
}
export -f eval_ace

# use xargs to run in parallel
N_PROCS=32
MODEL=$1 # e.g., ACE05-E/codex
find data/extraction/ace/inferred/$MODEL/* -maxdepth 1 -type d | cut -f5,6,7 -d '/' | xargs -P $N_PROCS -I {} bash -c 'eval_ace {}'

# aggregate results
set +e
set -x
function aggregate_evals() {
    MODEL=$1 # e.g., ACE05-E/codex

    # remove existing summary file
    rm data/extraction/ace/inferred/$MODEL/eval-headonly-summary.md
    rm data/extraction/ace/inferred/$MODEL/eval-headonly-summary.html

    python3 src/scripts/evaluation/aggregate-dir-eval.py \
        --dir data/extraction/ace/inferred/$MODEL \
        --filename test.jsonl.head_only.eval.md \
        --eval-id headonly \
        > data/extraction/ace/inferred/$MODEL/eval-headonly-summary.md

    pandoc \
        --standalone data/extraction/ace/inferred/$MODEL/eval-headonly-summary.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o data/extraction/ace/inferred/$MODEL/eval-headonly-summary.html \
        --metadata title="Evaluation Summary (Head Only)"
    # remove the intermediate markdown file
    rm data/extraction/ace/inferred/$MODEL/eval-headonly-summary.md

    # remove existing summary file
    rm data/extraction/ace/inferred/$MODEL/eval-coref-summary.md
    rm data/extraction/ace/inferred/$MODEL/eval-coref-summary.html

    python3 src/scripts/evaluation/aggregate-dir-eval.py \
        --dir data/extraction/ace/inferred/$MODEL \
        --filename test.jsonl.coref.eval.md \
        --eval-id coref \
        > data/extraction/ace/inferred/$MODEL/eval-coref-summary.md

    pandoc \
        --standalone data/extraction/ace/inferred/$MODEL/eval-coref-summary.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o data/extraction/ace/inferred/$MODEL/eval-coref-summary.html \
        --metadata title="Evaluation Summary (Head+Coref)"
    # remove the intermediate markdown file
    rm data/extraction/ace/inferred/$MODEL/eval-coref-summary.md

    # remove existing summary file
    rm data/extraction/ace/inferred/$MODEL/eval-span-summary.md
    rm data/extraction/ace/inferred/$MODEL/eval-span-summary.html

    python3 src/scripts/evaluation/aggregate-dir-eval.py \
        --dir data/extraction/ace/inferred/$MODEL \
        --filename test.jsonl.eval.md \
        --eval-id span \
        > data/extraction/ace/inferred/$MODEL/eval-span-summary.md

    pandoc \
        --standalone data/extraction/ace/inferred/$MODEL/eval-span-summary.md \
        --template src/tools/pandoc-toc-sidebar/toc-sidebar.html --toc \
        -o data/extraction/ace/inferred/$MODEL/eval-span-summary.html \
        --metadata title="Evaluation Summary (Span)"
    # remove the intermediate markdown file
    rm data/extraction/ace/inferred/$MODEL/eval-span-summary.md

}
echo "Aggregating results..."
aggregate_evals $MODEL
