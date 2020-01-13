# collusive-retweeters-TIST-2020
Analyzing and Detecting Collusive Users Involved in Blackmarket Retweeting Activities (ACM TIST 2020)

This is the code and the dataset for the paper titled 

>Analyzing and Detecting Collusive Users Involved in Blackmarket Retweeting Activities. *Udit Arora\*, Hridoy Sankar Dutta\*, Brihi Joshi, Aditya Chetan, Tanmoy Chakraborty*

submitted at [ACM Transactions on Intelligent Systems and Technology](https://dl.acm.org/journal/tist).

# Quick Start

## Requirements

- Python 3.6+
To install the dependencies used in the code, you can use the __requirements.txt__ file as follows -

```
pip install -r requirements.txt
```

## Dataset

```binary.csv``` and ```multi.csv``` contain the 50-dimensional user embeddings that give the best results for binary and multi-class classification of collusive/genuine users. Each line of the files represents a user, where the first 50 values represent the 50-dimension embedding, and the last value is the label.

```user_views.tsv.gz``` contains the user views corresponding to each user. Each row of the file contains tab separated values corresponding to the following for each user -

1. \[label 1 1 1 1 1 1\]: 'label' represents the annotated label for the user, and the six '1's represent the presence of the corresponding view for the user
2. \[AV1\]: Tweet2Vec view
3. \[AV2\]: SCoRe view
4. \[NV1\]: Retweet network view
5. \[NV2\]: Quote network view
6. \[NV3\]: Follower network view
7. \[NV4\]: Followee network view

### Labels

- label = 0: Genuine User
- label = 1: Normal Customer
- label = 2: Promotional Customer
- label = 3: Bot

## Running the code

### Embedding generation
Use https://github.com/abenton/wgcca with ```user_views.tsv``` as input to generate the multi-view user embeddings using WGCCA.

### Classification
Pick the type of dataset to be analyzed, and use the `run_classifier` method in ```classifiers.py``` to train a classifier and evaluate the performance.

# License 

Copyright (c) 2020 Udit Arora, Hridoy Sankar Dutta, Brihi Joshi, Aditya Chetan, Tanmoy Chakraborty

For license information, see [LICENSE](LICENSE) or http://mit-license.org
