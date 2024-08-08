This directory contains all the scraped, manually labeled, util and processed data for this project. 


It contains of 4 folders: 
```
data
├── manual
├── raw
├── results
└── databases
```

------------------------------

### `manual` 
This folder contains the schema and constituency tests that were manually developed for efficiently labelling data.
It also contains manual annotations of a small subset of the `raw` data using the defined schema and constituency tests.

Check the [manual README](manual/README.md) for further details.

### `raw`
This folder contains the raw data that was scraped from arXiv using the [arxiv](https://pypi.org/project/arxiv/) Python wrapper for different channels of research.

Check the [raw README](raw/README.md) for further details.

### `results`
This folder contains the processed results after running inference through the model. 
The model generates predictions of tagged concepts from the raw data and stores under folders within this folder.

Check the [results README](results/README.md) for further details.

### `databases`
This folder contains the SQL daatabases created for each pair of raw papers and the corresponding model predictions of tagged concepts. 

Check the [databases README](databases/README.md) for further details.