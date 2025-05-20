# Project Overview
This repository contains code for a comparative study of BERT and T5 models applied to Named Entity Recognition (NER) tasks. The study fine-tuned the models and evaluates the performance of both models on two tagging schemes: the original (7) scheme and a simplified (3) scheme.

# Files Description
This repository contains 4 scripts:
- full_ner_bert.py: This script implements NER using the BERT architecture with the original (7) tagging scheme, training and evaluating BERT for fine-grained entity recognition across specific entity categories.
- simple_ner_bert.py: This script implements NER based on BERT using a simplified (3) tagging scheme, where entity categories are merged into generic tags. It trains and evaluates the simplified BERT model for more coarse-grained entity recognition.
- full_ner_T5.py: This script implements NER based on T5 using the original (7) detailed tagging scheme. It trains and evaluates the T5 model to recognize entities with detailed category labels.
- simple_ner_T5: This script uses the T5 model with the simplified (3) tagging scheme, also can be used to train fine-tuned model and evaluate the performance on test set.

# Setup and Usage
## Environment
 - Python 3.8+
 - PyTorch 1.7+
 - Transformers library by Hugging Face
## Runing the scripts
Each Python script is self-contained and can be run as follows:
- ```
  python full_ner_t5.py
  ```
- ```
  python simple_ner_bert.py
  ```
- ```
  python simple_ner_t5.py
  ```
- ```
  python full_ner_bert.py
  ```
## Data
The code in each files downloads and processes the raw dataset into token-label pairs. First, it saves the processed data as a JSON file to avoid repeated downloads. Then, it loads this JSON file and formats the data into lists of [word, label] pairs for each sentence, preparing the train, dev, and test sets for model training.
- For example, in the full_ner_bert, the data processing code is
```
# Saving the data so that you don't have to redownload it each time.
with open('ner_data_dict.json', 'w', encoding='utf-8') as out:
    json.dump(data_dict, out, indent=2, ensure_ascii=False)
with open('ner_data_dict.json', 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
train_set = [[[word, label] for word, label in zip(words, labels)] for words, labels in data_dict['en_ewt']['train']]
dev_set = [[[word, label] for word, label in zip(words, labels)] for words, labels in data_dict['en_ewt']['dev']]
test_set = [[[word, label] for word, label in zip(words, labels)] for words, labels in data_dict['en_ewt']['test']]
test_set_ood =  [[[word, label] for word, label in zip(words, labels)] for words, labels in data_dict['en_pud']['test']]
```
- and the result for the first element in train_set is as follow:
```
[['Where', 'O'],
 ['in', 'O'],
 ['the', 'O'],
 ['world', 'O'],
 ['is', 'O'],
 ['Iguazu', 'B-LOC'],
 ['?', 'O']]
```
## Reproducibility Instructions
- The random seeds for all experiments are fixed to ensure reproducibility across runs.
- Training data is shuffled at the start of each epoch to ensure proper randomization.
- Early stopping based on labeled-span F1 score is used to avoid overfitting.
- Model checkpoints are saved at the best epoch for evaluation on test sets.

## Result
- full_ner_bert
```
=== In‐Domain Test Set ===
Unlabeled span score: 0.8539
Labeled   span score: 0.8217
Span‐level F1:        0.8161
 Token‐level F1 per tag:
  B-PER: 0.9341
  I-PER: 0.9281
  B-LOC: 0.8688
  I-LOC: 0.6102
  B-ORG: 0.7239
  I-ORG: 0.7117
  O: 0.9926
Token‐level Macro F1: 0.8242
```
```
=== Out‐of‐Domain Test Set ===
Unlabeled span score: 0.8316
Labeled   span score: 0.7684
Span‐level F1:        0.7694
Token‐level F1 per tag:
  B-PER: 0.9364
  I-PER: 0.9139
  B-LOC: 0.7940
  I-LOC: 0.7085
  B-ORG: 0.6098
  I-ORG: 0.7045
  O: 0.9921
Token‐level Macro F1: 0.8085
```
- simple_ner_bert
```
=== In-Domain Dev Set ===
Unlabeled span match:       0.8923
Labeled   span match:       0.8915
Span‐level labeled F1:      0.8700
Token‐level F1 per tag:
  B: 0.8960
  I: 0.8757
  O: 0.9927
Token‐level Macro F1 (7):   0.9215
```
```
=== Out-of-Domain Dev Set ===
Unlabeled span match:       0.8465
Labeled   span match:       0.8381
Span‐level labeled F1:      0.8417
Token‐level F1 per tag:
  B: 0.8902
  I: 0.8316
  O: 0.9922
Token‐level Macro F1 (7):   0.9047
```
- full_ner_T5
```
Validate: 100%
 2077/2077 [02:32<00:00,  8.21it/s]

=== In-Domain Test ===
Unlab span match: 0.8364
Labl span match:  0.7904
Span‐level F1:    0.8206
Per‐label F1:
   B-LOC: 0.8527
   B-ORG: 0.6964
   B-PER: 0.9265
   I-LOC: 0.5926
   I-ORG: 0.6732
   I-PER: 0.8984
       O: 0.9927
Macro‐F1:         0.8047
```
```
Validate: 100%
 1000/1000 [01:28<00:00,  9.82it/s]
=== Out-of-Domain Test ===
Unlab span match: 0.8316
Labl span match:  0.7498
Span‐level F1:    0.7803
Per‐label F1:
   B-LOC: 0.7880
   B-ORG: 0.5989
   B-PER: 0.9147
   I-LOC: 0.7039
   I-ORG: 0.7333
   I-PER: 0.8743
       O: 0.9927
Macro‐F1:         0.8008
```
- simple_ner_T5
```
Validate: 100%
 2077/2077 [01:55<00:00, 13.98it/s]

=== In-Domain Test ===
Unlab span match: 0.8524
Labl span match:  0.8483
Span‐level F1:    0.8441
Per‐label F1:
       B: 0.8861
       I: 0.8317
       O: 0.9915
Macro‐F1:         0.9031
```
```
Validate: 100%
 1000/1000 [00:57<00:00, 15.53it/s]

=== Out-of-Domain Test ===
Unlab span match: 0.8763
Labl span match:  0.8763
Span‐level F1:    0.8698
Per‐label F1:
       B: 0.9067
       I: 0.8386
       O: 0.9928
Macro‐F1:         0.9127
```
