- This folder contains all the SQL databases for the different processed data along with their raw data.

- The databases are named after the arXiv category and the format of the generated data.

Each file in this folder is a database containing 2 tables:
- **papers**
    
    The papers data from the `raw` folder that was fed to the model.
    
    SCHEMA:
    - paper_id TEXT PRIMARY KEY,
    - abstract TEXT,
    - authors TEXT,
    - primary_category TEXT,
    - url TEXT,
    - updated_on TEXT,
    - sentence_count INTEGER

- **predictions**
    
    The corresponding model generations stored in the `results` folder.

    SCHEMA:
    - id INTEGER PRIMARY KEY AUTOINCREMENT,
    - paper_id TEXT,
    - sentence_index INTEGER,
    - tag_type TEXT,
    - concept TEXT,
    - FOREIGN KEY (paper_id) REFERENCES papers(paper_id)


To query any database, open SQLite in your terminal and specify the database name. 