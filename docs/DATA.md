# Data

## Process `ACE05` data into `oneie` format

First, obtaining raw ACE 2005 corpus from [here](https://catalog.ldc.upenn.edu/LDC2006T06) and put it to `data/extraction/ace/raw/ace_2005_td_v7`.

Then, download OneIE from [here](http://blender.cs.illinois.edu/software/oneie/) and unzip it to `data/extraction/ace/raw/oneie_v0.4.8`.

```bash
cd data/extraction/ace/raw/
git clone https://github.com/dwadden/dygiepp.git

# switch to a different env due to compatibility issue, see https://github.com/dwadden/dygiepp#creating-the-dataset-1
conda create --name ace-event-preprocess python=3.7
conda activate ace-event-preprocess
pip install -r scripts/data/ace-event/requirements.txt
python -m spacy download en_core_web_sm

# Covert data using dygiepp
cd dygiepp 
bash ./scripts/data/ace-event/collect_ace_event.sh ../ace_2005_td_v7
python ./scripts/data/ace-event/parse_ace_event.py evt-entity-coref+pronouns --include_event_coreference --include_pronouns --include_entity_coreference
```

Then switch to current project environment (`conda activate code4struct`), and convert `dygiepp` format into `oneie` format.

```bash
mkdir -p data/extraction/ace/interim/ACE05-E/split

python3 data/extraction/ace/raw/oneie_v0.4.8/preprocessing/process_dygiepp.py \
    --input data/extraction/ace/raw/dygiepp/data/ace-event/processed-data/evt-entity-coref+pronouns/json/train.json \
    --output data/extraction/ace/interim/ACE05-E/split/train.oneie.json

python3 data/extraction/ace/raw/oneie_v0.4.8/preprocessing/process_dygiepp.py \
    --input data/extraction/ace/raw/dygiepp/data/ace-event/processed-data/evt-entity-coref+pronouns/json/dev.json \
    --output data/extraction/ace/interim/ACE05-E/split/dev.oneie.json

python3 data/extraction/ace/raw/oneie_v0.4.8/preprocessing/process_dygiepp.py \
    --input data/extraction/ace/raw/dygiepp/data/ace-event/processed-data/evt-entity-coref+pronouns/json/test.json \
    --output data/extraction/ace/interim/ACE05-E/split/test.oneie.json
```

## Build ontology templates for `ACE05-E`

1. Build LDC base entities: It will create one entity mapping (in `.json`) AND a code file with converted entity structure (`.py`).

```bash
python3 src/scripts/data/build-ldc-base-entities.py data/ontology/ldc/base_entities/base_entities
```

2. Parse `ACE05` event: Using ACE ontology about event and role constraints (role must be a subset of base entities) to build a mapping (`.json`) that map ACE05 event type to code (with context, i.e. base entities used by an event), as well as a code file (`.py`) that contains all events for reading.

```bash
PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/parse_ACE_events.py \
    --ace-roles-filepath data/ontology/ace/raw/event_role_ACE.json \
    --ace-entity-filepath data/ontology/ace/raw/event_role_entity_ACE.json \
    --base-entity-filepath data/ontology/ldc/base_entities/base_entities.json \
    --output-filepath data/ontology/ace/processed/ace_roles+hierarchy \
    --add-hierarchy

# w/ pure text (for text-based GPT-3 prompt)
PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/parse_ACE_events.py \
    --ace-roles-filepath data/ontology/ace/raw/event_role_ACE.json \
    --ace-entity-filepath data/ontology/ace/raw/event_role_entity_ACE.json \
    --base-entity-filepath data/ontology/ldc/base_entities/base_entities.json \
    --output-filepath data/ontology/ace/processed/ace_roles+pure_text \
    --pure-text-prompt
```

3. Convert ACE05 dataset (in this case, the test set) to code generation ready format, using previously converted ACE ontology.

### `Code4Struct`
```bash
# This process ACE05 to code with parent event and child event definitions
PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/process_ACE_dataset.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+hierarchy.json \
    --input-filepath data/extraction/ace/interim/ACE05-E/split/test.oneie.json \
    --output-filedir data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/ \
    --mark-trigger

PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/process_ACE_dataset.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+hierarchy.json \
    --input-filepath data/extraction/ace/interim/ACE05-E/split/train.oneie.json \
    --output-filedir data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/ \
    --mark-trigger
```

### Text-based Prompt

```bash
# v5.1 + trigger
PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/process_ACE_dataset.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+pure_text.json \
    --input-filepath data/extraction/ace/interim/ACE05-E/split/test.oneie.json \
    --output-filedir data/extraction/ace/processed/ACE05-E/v5.1-baseline+puretext+trigger/ \
    --pure-text-prompt --mark-trigger

PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/process_ACE_dataset.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+pure_text.json \
    --input-filepath data/extraction/ace/interim/ACE05-E/split/train.oneie.json \
    --output-filedir data/extraction/ace/processed/ACE05-E/v5.1-baseline+puretext+trigger/ \
    --pure-text-prompt --mark-trigger
```

**`Cross-sibling Transfer`**
```bash
# This augment training examples from sibling event types for in-context learning
# 10-shot
mkdir -p data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/
cp data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/test.jsonl data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/test.preprocess.jsonl
cp data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/train.jsonl data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/train.preprocess.jsonl

PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/augment_examples_to_processed_ACE.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+hierarchy.json \
    --input-filepath data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/test.preprocess.jsonl \
    --input-train-filepath data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/train.preprocess.jsonl \
    --output-filedir data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+10hierarchyexample+siblingonly/ \
    --hierarchy-split-filepath data/extraction/ace/interim/ACE05-E/hierarchy-split/train_test_hierarchy.json \
    --n-hierarchy-incontext-examples 10 \
    --only-sibling-examples

# 1-shot
mkdir -p data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/
cp data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/test.jsonl data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/test.preprocess.jsonl
cp data/extraction/ace/processed/ACE05-E/v6.4-baseline+trigger+hierarchy/train.jsonl data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/train.preprocess.jsonl

PYTHONPATH=`pwd`:$PYTHONPATH python3 src/scripts/data/augment_examples_to_processed_ACE.py \
    --parsed-ace-roles data/ontology/ace/processed/ace_roles+hierarchy.json \
    --input-filepath data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/test.preprocess.jsonl \
    --input-train-filepath data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/train.preprocess.jsonl \
    --output-filedir data/extraction/ace/processed/ACE05-E/v9.2-baseline+trigger+1hierarchyexample+siblingonly/ \
    --hierarchy-split-filepath data/extraction/ace/interim/ACE05-E/hierarchy-split/train_test_hierarchy.json \
    --n-hierarchy-incontext-examples 1 \
    --only-sibling-examples
```
