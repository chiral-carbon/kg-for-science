```
manual
├── constituency_tests.json
├── human_annotations.jsonl
└── schema.json
```

----------------------------

- `schema.json` contains the schema that informs the definition of the different concepts in a scientific article. 
    
    This schema was developed by the project's authors through a few iterations of discussion. 
    
    The schema was:
    - Used to guide human annotators in labeling a subset of raw data
    - passed to the language model with instructions for annotating the raw data during in-context learning


    Defining the schema ensures consistency and standardization in identifying and categorizing scientific concepts for both humans and the language model. It provides a common framework for comparing and aggregating data across multiple papers.


- `constituency_tests.json` contains the tests for constituency that were defined for clarifying any follow-up questions one may have about the schema and the information extraction/tagging process.

    The constituency tests were also developed by the project's authors and is useful for:
    - Ambiguity resolution during the manual annotations phase: We may encounter concepts that may be categorized as more than tag based on only the schema definitions
    - Ambiguity resolution for the model while generating tagged concepts: Though not utilized for now, it can potentially be used as a chain-of-thought or reasoning tool and passed to the language model along with instructions in order to have refined, less ambiguous tag predictions

    It is currently mainly used as a reference points for manual annotators when they encounter ambiguity. 


- `human_annotations.jsonl` contains 20 manually annotated paper titles and abstracts from the `astro-ph` arXiv channel stored in json lines. 
    
    - They contain annotations done by the project's authors based on the defined schema. Constituency tests help in resolving ambiguity. 
    - The annotations are used for prompt optimization by dividing it into `train` and `dev` sets:
        - the first 3 papers are used as the `train` set for few-shot examples and the rest are used as the `dev` set and passed to the language model for generating tags
        - across different sweeps of hyperparameters on the `train` set prefixes, the precision and recall of the predicted and grounth truth tags in the `dev` set are measured to identify the best hyperparameters
    - These hyperparameters are then applied to the prefixes from the `train` set when generating tags for the `test` set--which is a large raw dataset from arXiv.