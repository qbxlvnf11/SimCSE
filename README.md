Contents
=============

### - SimCSE (Simple Contrastive Learning of Sentence Embeddings) Sample Code
* Generating Unsupervised & Supervised SimCSE Data
* SimCSE Train

Generating Unsupervised & Supervised SimCSE Data
=============

```
python gen_simcse_data.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} --output_unsupervised {unsupervised train data path} --output_supervised {supervised train data path}
```


