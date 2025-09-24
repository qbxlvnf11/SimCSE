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
   
  1. [Baseline](gen_data/simcse_data_gen_basic.py)

  * Unsupervised learning data: converting all columns in a row to a text
  * Mapping method of supervised learning data:
    * For each anchor, create anchor–positive pairs (number of positive columns pairs) using each positive column.
    * Attach one random negative to each pair.

```
python gen_data/gen_simcse_data.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} --output_unsupervised {unsupervised train data save path} --output_supervised {supervised train data save path}
```

  2. [Domainwise Version](gen_data/simcse_data_gen_domainwise.py)

    * Same as baseline, but with domain-based data separation. (Each domain gets its own dataset file.)

```
python gen_data/gen_simcse_data.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} ----domain_col {domain column} --output_dir {unsupervised, supervised train data save folder path}
```

  3. [Multi-Negative with Positive Fusion Version](gen_data/simcse_data_gen_domainwise.py)

    * Unsupervised learning data: converting all columns in a row to a text
    * Mapping method of supervised learning data:
      * All positive values are fused into a single string
          * E.g. "pos column1: xxx, pos column2: yyy, ..."
      * For each anchor, attach multiple hard negatives (default: 5).
      * One row per anchor
      
```
python gen_data/simcse_data_gen_fusion_multineg.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} ----domain_col {domain column} --output_dir {unsupervised, supervised train data save folder path} --num_negatives {num of negatives}
```
