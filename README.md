# KG For Science

This is a WIP that builds knowledge graphs to extract structured information from scientific publications, datasets and articles. 

We want to cover all of "science", and perform semantic search and interact with the tool, and be able to perform RAG. 

We use Llama-3 70B for structured information extraction. 

## Installation 

Clone and navigate to the repository:
```
git clone https://github.com/chiral-carbon/kg-for-science.git
cd kg-for-science
```
Create a conda or virtual environment:
```
conda env create -f environment.yml
```
Activate the environment:
```
conda activate kg4s
```
~~Download the spacy model for paragraph splitting:~~
~~python -m spacy download en_core_web_sm~~

Set up code formatting and pre-commit hooks:
```
black .
pre-commit install
```

## Running the tool

1. Run `scripts/collect_data.py` to download papers for arXiv:
```
python scripts/data_collect.py --max_results 1000 --search_query astro-ph --sort_by 'last_updated_date' --sort_order desc 
```

These are the default arguments, you can modify them to specify the arxiv channel, number of papers and order of search. The data is stored in the `data` directory.

2. Then run `main.py` to call Llama-3 70B and perform extractions on the downloaded papers using Slurm jobs:
```
sbatch run.sh
```
You can modify the arguments passed to `main.py` as required (evaluation on dev set or extracting data with a new dataset).

You can view the options by running `python main.py --help`:
```
Usage: main.py [OPTIONS]

Options:
  --kind TEXT              Specify the kind of prompt input: json (default) or
                           readable
  --runtype [new|eval]     Specify the type of run: new or eval (default)
  --data TEXT              Specify the directory of the data if running on new
                           data
  --sweep                  Run sweeps
  --sweep_config TEXT      Sweep configuration file
  --load_best_config TEXT  Load the best configuration from a file
  --help                   Show this message and exit.
```

We use 2 A100 80GB GPUs to perform extractions with Llama-3 70B. You can choose a different model if limited by memory and GPU. 

The current best performance on the dev set:

| Metric | kind=json | kind=readable |
|--------|-----------|---------------|
| precision | 0.4329 | 0.4364 |
| recall | 0.3974 | 0.3110 |
| f1 | 0.4144 | 0.3632 |
| union_precision | 0.5864 | 0.6242 |
| union_recall | 0.5216 | 0.4459 |
| union_f1 | 0.5521 | 0.5202 |
| avg_time_per_sentence | 4.0315 | 2.7584 |
| total_time | 463.6508 | 317.2468 |

3. To create a SQLite3 database of the predictions, run:
```
python scripts/create_db.py --data_path <path to the jsonl file with data> --predictions_path <path to the predictions.json file>
```
this creates a database in the `tables` directory.
```
Usage: create_db.py [OPTIONS]

Options:
  --data_path TEXT         Path to the data file containing the papers
                           information.
  --predictions_path TEXT  Path to the predictions file.
  --db_name TEXT           Name of the database to create.
  --force                  Force overwrite if database already exists
  --help                   Show this message and exit.
  ```

4. To run a SQL query search over the created database as a Gradio interface, run:
```
python scripts/run_db_interface.py
```

## Relevant Resources for Reference
### Tools
- Nomic AI's Atlas has beautiful and interactive visualizations and also provides an embedding API for visualizing knowledge graphs | [Site](https://atlas.nomic.ai/) | [GitHub](https://github.com/nomic-ai/nomic)
- Instagraph is a good starting point for KG viz. They generate the knowldge graph (nodes and edges) given a knowledge base using GPT 3.5 | [Site](https://instagraph.ai) | [GitHub](https://github.com/yoheinakajima/instagraph)
- The Monarch Initiative has a neat interface for phenotype/gene/disease knowledge discovery | [Site](https://next.monarchinitiative.org) | [GitHub](https://github.com/monarch-initiative)

### Research Papers
- Knowledge Graph in Astronomical Research with Large Language Models: Quantifying Driving Forces in Interdisciplinary Scientific Discovery | [arXiv](https://arxiv.org/pdf/2406.01391)
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models | [arXiv](https://arxiv.org/pdf/2308.09687)