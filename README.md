Contents
=============

* Generating Unsupervised & Supervised Train Data of Text Embedding Model
* Text Embedding Model Fine-Tuning Methods: SimCSE, Triplet Loss etc.

Run
=============

### - Generating Unsupervised & Supervised Data for Fine-Tuning Embedding Model

<details>
<summary>Data Generation Methods</summary>

  * Data Generation Scripts
    * This repository contains scripts for generating training datasets.
    * Each script creates both unsupervised samples and supervised samples (anchor, positive, negative) from an input CSV file.
    * They differ in how they construct anchor–positive–negative pairs, how the data is split, and how negatives are sampled.
   
  1. [Baseline](gen_data/text_embedder_fine_tuning_data_gen_basic.py)

  * Unsupervised learning data: converting all columns in a row to a text
  * Mapping method of supervised learning data:
    * For each anchor, create anchor–positive pairs (number of positive columns pairs) using each positive column.
    * Attach one random negative to each pair.

```
python gen_data/text_embedder_fine_tuning_data_gen_basic.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} --output_unsupervised {unsupervised train data save path} --output_supervised {supervised train data save path}
```

  2. [Domainwise Version](gen_data/text_embedder_fine_tuning_data_gen_domainwise.py)

  * Same as baseline, but with domain-based data separation. (Each domain gets its own dataset file.)

```
python gen_data/text_embedder_fine_tuning_data_gen_domainwise.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} ----domain_col {domain column} --output_dir {unsupervised, supervised train data save folder path}
```

  3. [Multi-Negative with Positive Fusion Version](gen_data/text_embedder_fine_tuning_data_gen_fusion_multineg.py)

  * Unsupervised learning data: converting all columns in a row to a text
  * Mapping method of supervised learning data:
    * All positive values are fused into a single string
      * E.g. "pos column1: xxx, pos column2: yyy, ..."
  * For each anchor, attach multiple hard negatives (default: 5).
  * One row per anchor

```
python gen_data/text_embedder_fine_tuning_data_gen_fusion_multineg.py --data_path {csv data path} --encoding {encoding} --desc_col {anchor column} --category_col {hard negative column} --positive_cols {positive column1, ...} ----domain_col {domain column} --output_dir {unsupervised, supervised train data save folder path} --num_negatives {num of negatives}
```

</details>

### - Embedding Gemma (300M) Fine-Tuning Sample Code

<details>
<summary>Fine-Tuning Process</summary>

* Run env

```
conda create --name gemma-embedding python=3.10 -y
conda info --envs
conda activate gemma-embedding
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
pip install -r embedding_gemma_requirements.txt
pip install --upgrade accelerate transformers
```

* Fine-tuning
  * enter the Huggingface Token (huggingface_token) in the '.env'

```
export CUDA_VISIBLE_DEVICES=0
python embedding_gemma_fine_tuning_test.py
```

* Results

```
- Query: I want to start a tax-free installment investment, what should I do?
Document: Opening a NISA Account -> 🤖 Score: 0.403728
Document: Opening a Regular Savings Account -> 🤖 Score: 0.329424
Document: Home Loan Application Guide -> 🤖 Score: 0.108175
```

</details>

Reference
=============

#### - [SimCSE (Simple Contrastive Learning of Sentence Embeddings)](https://aclanthology.org/2021.emnlp-main.552) 

#### - [EmbeddingGemma Fine-Tuning](https://ai.google.dev/gemma/docs/embeddinggemma/fine-tuning-embeddinggemma-with-sentence-transformers?hl=ko) 

#### - [MultipleNegativesRankingLoss, TripletLoss](https://blog.naver.com/qbxlvnf11/224034908636) 


Author
=============

#### - [LinkedIn](https://www.linkedin.com/in/taeyong-kong-016bb2154)

#### - [Blog](https://blog.naver.com/qbxlvnf11)

#### - Email: qbxlvnf11@google.com, qbxlvnf11@naver.com

