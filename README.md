Contents
=============

### - [SimCSE (Simple Contrastive Learning of Sentence Embeddings)](https://aclanthology.org/2021.emnlp-main.552) Sample Code
* Generating Unsupervised & Supervised SimCSE Data
* SimCSE Train

Generating Unsupervised & Supervised SimCSE Data
=============

  * SimCSE Data Generation Scripts
    * This repository contains three scripts for generating SimCSE training datasets.
    * Each script creates both unsupervised samples and supervised samples (anchor, positive, negative) from an input CSV file.
    * They differ in how they construct anchor–positive–negative pairs, how the data is split, and how negatives are sampled.
   
  1. [Baseline](gen_simcse_data.py)

```
python gen_simcse_data.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} --output_unsupervised {unsupervised train data path} --output_supervised {supervised train data path}
```

  2. 
