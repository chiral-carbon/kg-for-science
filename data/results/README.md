This folder contains all the resulting model generations for tagged concepts once the raw data is passed to the model. 

It stores data folders named after:
- the mode of the run: `eval` when running on dev set OR `new` when running the model on the test set 
- the order of selecting few-shot examples from the train set: "random", "first", "last", "middle", "distributed", "longest", "shortest"
- the format of outputs produced: `json` OR `readable`
- a UUID from randomized alphanumeric symbols


Refer to [manual/README.md](../manual/README.md) to undestand the `train`, `dev` and `test` sets.


Files in each folder in `results`:

1. `metrics.json`: contains the metrics of the run. For `new` mode, only the average and total time is reported as there is ground truth to measure comparison metrics. 
2. `predicted_responses.txt`: contains the raw model responses in text format which is then processed
3. `predictions.json`: contains a JSON object of all the model predictions at the sentence level for each sentence in the test or dev set, extracted from the model response text
4. `prompts.txt`: contains all the prompts (few-shot examples prefix + input sentences from paper titles and abstracts), model responses, the extracted predictions from model responses and the true tags
5. `logs/log.txt`: the logger outputs