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
conda create --name kg4s python=3.11
```
Activate the environment:
```
conda activate kg4s
```
Install the dependencies:
```
pip install -r requirements.txt
```
~~Download the spacy model for paragraph splitting:~~
~~python -m spacy download en_core_web_sm~~

Set up code formatting and pre-commit hooks:
```
black .
pre-commit install
```

## Running the tool

Then, run `scripts/data_collect.py` to download papers for arXiv:
```
python scripts/data_collect.py --max_results 1000 --search_query astro-ph --sort_by 'last_updated_date' --sort_order desc 
```

These are the default arguments, you can modify them to specify the arxiv channel, number of papers and order of search. The data is stored in the `data` directory.

Then run `main.py` to call Llama-3 70B and perform extractions on the downloaded papers:
```
python main.py
```

We use 2 A100 80GB GPUs to perform extractions with Llama-3 70B. You can choose a different model if limited by memory and GPU. 

## Relevant Resources for Reference:
### Tools
- Nomic AI's Atlas has beautiful and interactive visualizations and also provides an embedding API for visualizing knowledge graphs | [Site](https://atlas.nomic.ai/) | [GitHub](https://github.com/nomic-ai/nomic)
- Instagraph is a good starting point for KG viz. They generate the knowldge graph (nodes and edges) given a knowledge base using GPT 3.5 | [Site](https://instagraph.ai) | [GitHub](https://github.com/yoheinakajima/instagraph)
- The Monarch Initiative has a neat interface for phenotype/gene/disease knowledge discovery | [Site](https://next.monarchinitiative.org) | [GitHub](https://github.com/monarch-initiative)

### Research Papers
- Graph of Thoughts: Solving Elaborate Problems with Large Language Models | [arXiv](https://arxiv.org/pdf/2308.09687)
- Knowledge Graph in Astronomical Research with Large Language Models: Quantifying Driving Forces in Interdisciplinary Scientific Discovery | [arXiv](https://arxiv.org/pdf/2406.01391)