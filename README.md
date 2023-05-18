# Code4Struct: Code Generation for Few-Shot Structured Prediction from Natural Language

Official repo for paper [Code4Struct: Code Generation for Few-Shot Structured Prediction from Natural Language](https://arxiv.org/abs/2210.12810).


## Environment Setup

```
conda env create -f environment.yml
conda activate code4struct
```

## Data

Please refer to [docs/DATA.md](docs/DATA.md) for detailed instructions.

## Inference

You will need to obtain your API key from [here](https://beta.openai.com/account/api-keys).

```bash
export OPENAI_API_KEY="YOUR_API_KEY_HERE"
./src/scripts/model/batch-exp.sh
```

## Evaluate generated results
```bash
./src/scripts/evaluation/eval-all-ace.sh ACE05-E/codex
```

Evaluation result for each experiment run will be saved to the corresponding output_dir (e.g., `data/extraction/ace/inferred/ACE05-E/codex/v6.4-baseline+trigger+hierarchy-50shot-n1-t0.0`).

## Visualize Evaluation Result

You can also visualize evaluation result using `localhost:8000` by running the following: 
```bash
streamlit run --server.port 8000 src/scripts/evaluation/streamlit-viz.py
```

## Citation

```
@article{wang2022code4struct,
  title={Code4Struct: Code Generation for Few-Shot Structured Prediction from Natural Language},
  author={Wang, Xingyao and Li, Sha and Ji, Heng},
  journal={arXiv preprint arXiv:2210.12810},
  year={2022}
}
```
