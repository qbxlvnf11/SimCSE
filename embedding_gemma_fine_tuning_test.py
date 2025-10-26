## Setup
from huggingface_hub import login
import os
from dotenv import load_dotenv
import time

import torch
from torch.utils.data import DataLoader
from datasets import Dataset

from transformers import TrainerCallback
from sentence_transformers import InputExample, SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import TripletLoss, MultipleNegativesRankingLoss, AnglELoss, TripletDistanceMetric
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from train.call_back import SaveBestModelCallback
from train.evaluator import SampleMarginEvaluator, SampleCombinedMarginEvaluator, get_scores
from train.utils import model_checksum, set_seeds, make_seed, add_prompt

# --- Environment and Login ---
print('\n=====================  Environments & Params =====================')
load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token:
    login(token=huggingface_token)
    print("[ Hugging Face login successful! ]")
else:
    print("[ Hugging Face token not found. ]")

# --- Seed ---
# seed = 42
seed = make_seed(auto=False)
set_seeds(seed)

# --- Train Params ---
num_train_epochs = 20
per_device_train_batch_size = 3 # 2, 3, 5
learning_rate = 5e-5 #1e-5
logging_steps = 10
warmup_ratio = 0.1
# new_dropout_rate = 0.5 #0.5 #0.1
loss_name = 'TripletLoss' # 'TripletLoss', 'AnglELoss', 'MultipleNegativesRankingLoss'
## 마진을 키우고 싶으면 scale을 올리고, 불안정하면 낮추세요.
ranking_loss_scale = 30 # 10, 20, 30, 40 (default=20)
angle_loss_scale = 30 # 10, 20, 30, 40 (default=20)
triplet_margin = 0.7 # 0.2, 0.3, 0.5, 0.7, 1.0
# task_name = "STS" #"STS" # Semantic Textual Similarity
print(f" - seed: {seed}") 
print(f" - num_train_epochs: {num_train_epochs}")
print(f" - per_device_train_batch_size: {per_device_train_batch_size}")
print(f" - learning_rate: {learning_rate}")
print(f" - loss_name: {loss_name}")
print(f" - angle_loss_scale: {angle_loss_scale}")
print(f" - ranking_loss_scale: {ranking_loss_scale}")
print(f" - triplet_margin: {triplet_margin}")
# print(f" - task_name: {task_name}")
print("==========================================")

# --- Model Loading ---
print('\n=====================  Model Setup =====================')
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id).to(device=device)
print(f" - Device: {model.device}")

# --- Model ---
transformer_model = model[0].auto_model
transformer_config = transformer_model.config
print(f" - transformer_config: {transformer_config}")
# print(f" - Default attention_dropout: {transformer_config.attention_dropout}")

# dropout_layer_num = 0
# if new_dropout_rate != 0.0:
#     # transformer_config.attention_dropout = new_dropout_rate

#     # transformer_model = model[0].auto_model
#     # transformer_config = transformer_model.config
#     # print(f" - Attention dropout rate has been changed to: {transformer_config.attention_dropout}")

#     for name, module in model.named_modules():
#         ## Transformer 내부의 Dropout 레이어만 찾아서 변경
#         if isinstance(module, torch.nn.Dropout):
#             dropout_layer_num += 1
#             module.p = new_dropout_rate

#     print(" - All Dropout layers updated to:", new_dropout_rate, f"(dropout_layer_num: {dropout_layer_num})")
# # print(f" - Default attention_dropout: {transformer_config.attention_dropout}")
print("==========================================")

## --- Sample Dataset -
print('\n===================== Sample Dataset =====================')
print(" - Prompt keys:", list(model.prompts.keys()))
q_prompt = model.prompts["Retrieval-query"] #model.prompts["query"] #model.prompts["Retrieval-query"]
d_prompt = model.prompts["Retrieval-document"] #model.prompts["document"] #model.prompts["Retrieval-document"]
print(" - Query prompt:", q_prompt)
print(" - Doc prompt:", d_prompt)

# dataset_list = [
#     ["How do I open a NISA account?", "What is the procedure for starting a new tax-free investment account?", "I want to check the balance of my regular savings account."],
#     ["Are there fees for making an early repayment on a home loan?", "If I pay back my house loan early, will there be any costs?", "What is the management fee for this investment trust?"],
#     ["What is the coverage for medical insurance?", "Tell me about the benefits of the health insurance plan.", "What is the cancellation policy for my life insurance?"],
# ]
dataset_list = [
    # Category 1: ISA (Individual Savings Account)
    ["How do I open an ISA account?",
     "What is the procedure for starting an Individual Savings Account?",
     "I want to check the balance of my regular savings account."], # Hard Negative
    ["What are the main benefits of an ISA account?",
     "Explain the advantages and features of ISA products.",
     "How is it different from a 401(k) or a personal pension plan?"], # Very Hard Negative
    ["Is there a limit to how much I can deposit into my ISA?",
     "Tell me about the ISA contribution limits.",
     "What is the tax deduction limit for credit card spending?"], # Hard Negative
    ["What kinds of products can I invest in with a Stocks & Shares ISA?",
     "I'm curious about the types of financial products available in an ISA.",
     "What is the stock market forecast for today?"], # Hard Negative
    ["How do I extend my ISA once it matures?",
     "I need information on renewing my matured ISA account.",
     "Can you recommend some popular restaurants?"], # Easy Negative

    # Category 2: Home Loan / Mortgage
    ["Are there fees for making an early repayment on a home loan?",
     "What are the conditions for early repayment penalties on a mortgage?",
     "What about the early repayment fees for a rent deposit loan?"], # Hard Negative
    ["What documents do I need to apply for a mortgage?",
     "Please list the required paperwork for an online mortgage application.",
     "What documents are needed to apply for a credit card?"], # Hard Negative
    ["How are the LTV and DSR limits for a home loan calculated?",
     "I'm curious about the maximum mortgage amount I can get.",
     "I'd like to check my personal credit loan limit."], # Hard Negative
    ["What are the pros and cons of a variable-rate mortgage?",
     "I'm debating between a fixed-rate and a variable-rate home loan.",
     "Can you compare car insurance quotes for me?"], # Easy Negative

    # Category 3: Medical Insurance
    ["What is covered by this medical insurance plan?",
     "Tell me about the benefits included in the health insurance policy.",
     "I'm interested in the death benefit coverage of a life insurance policy."], # Hard Negative
    ["How do I file a claim for my medical insurance? What documents are needed?",
     "What is the procedure and paperwork for getting reimbursed for hospital bills?",
     "I need to handle the insurance claim for a car accident."], # Hard Negative
    ["Is dental care covered by my health insurance?",
     "Does the medical plan include dental treatments?",
     "What illnesses are covered by pet insurance?"], # Hard Negative
    ["What is the co-payment for the 4th generation medical insurance?",
     "Explain the out-of-pocket expense ratio for this health plan.",
     "What's the deductible for traveler's insurance?"], # Hard Negative

    # Category 4: Credit Card (New Category)
    ["What are the requirements to get a credit card? Can I get one with no income?",
     "I'm a college student, am I eligible for a credit card?",
     "What are the requirements for getting a debit card?"], # Very Hard Negative
    ["Can you recommend a good credit card for earning airline miles?",
     "Please compare credit cards with high reward point rates.",
     "How do I apply for a public transit discount card?"], # Hard Negative
    ["Is there a way to get the annual fee waived for my credit card?",
     "I want to check if I've met the conditions for an annual fee waiver.",
     "How can I get bank transfer fees waived?"], # Hard Negative
    ["Can I pay off my credit card bill early?",
     "How do I apply for an early payment on my card balance?",
     "I want to increase the spending limit for my mobile phone payments."], # Hard Negative
    ["I'm looking for a credit card to use overseas.",
     "Are there any credit cards with no foreign transaction fees?",
     "Can you help me plan my trip abroad?"], # Easy Negative
]

if loss_name == 'TripletLoss':

    anchor_list = []
    positive_list = []
    negative_list = []
    for a, p, n in dataset_list:
        anchor_list.append(add_prompt(a, q_prompt))
        positive_list.append(add_prompt(p, d_prompt))
        negative_list.append(add_prompt(n, d_prompt))

    train_dataset = Dataset.from_dict({ \
        "anchor":anchor_list, \
        "positive":positive_list,\
        "negative":negative_list,\
        })

elif loss_name == 'MultipleNegativesRankingLoss':
    # data_as_dicts = [{"anchor": row[0], "positive": row[1], "negative": row[2]} for row in dataset_list]
    # train_dataset = Dataset.from_list(data_as_dicts)

    pairs = []
    for anchor, positive, _ in dataset_list:  # negative는 MNR에서 직접 쓰지 않음
        a = add_prompt(anchor, q_prompt)      # 예: Retrieval-query 또는 query 프롬프트
        p = add_prompt(positive, d_prompt)    # 예: Retrieval-document 또는 document 프롬프트
        pairs.append({"sentence1": a, "sentence2": p})
    train_dataset = Dataset.from_list(pairs)

elif loss_name == 'AnglELoss':
    
    pairs = []
    soft_pos_score = 0.95 # 0.9 # 0.95
    soft_neg_score = 0.05 # 0.1 # 0.05
    print(' - soft_pos_score:', soft_pos_score)
    print(' - soft_neg_score:', soft_neg_score)
    
    for anchor, positive, negative in dataset_list:

        anchor = add_prompt(anchor, q_prompt)
        positive = add_prompt(positive, d_prompt)
        negative = add_prompt(negative, d_prompt)

        ## Soft
        pairs.append({"sentence1": anchor, "sentence2": positive, "score": soft_pos_score}) # 0.9 # 0.95
        pairs.append({"sentence1": anchor, "sentence2": negative, "score": soft_neg_score}) # 0.1 # 0.05
        ## Hard
        # pairs.append({"sentence1": anchor, "sentence2": positive, "score": 1.0})
        # pairs.append({"sentence1": anchor, "sentence2": negative, "score": 0.0})

    train_dataset = Dataset.from_list(pairs)

print('Train dataset:', train_dataset)
print("==========================================")

print('\n------------------ Before Training ------------------')
print('\n===================== Training =====================')
# --- Define Query and Documents for Evaluation ---
query = "I want to start a tax-free installment investment, what should I do?"
documents = ["Opening a NISA Account", "Opening a Regular Savings Account", "Home Loan Application Guide"]
get_scores(model, query, documents, q_prompt, d_prompt)

## --- Training ---
model.train()

trainable = 0
total = 0
for name, param in model.named_parameters():
    num = param.numel()
    total += num
    if param.requires_grad:
        trainable += num
    else:
        print(f"   --- FROZEN: {name} ({num} params)")

print(f" - Trainable params: {trainable} / {total} ({trainable/total:.2%})")
# print(' - model.prompts[task_name]:', model.prompts[task_name])
print("==========================================")

## Initialize loss with the correct model variable
if loss_name == 'TripletLoss':
    loss = TripletLoss(model, triplet_margin=triplet_margin, \
        distance_metric=TripletDistanceMetric.COSINE)
if loss_name == 'MultipleNegativesRankingLoss':
    loss = MultipleNegativesRankingLoss(model, scale=ranking_loss_scale)
elif loss_name == 'AnglELoss':
    loss = AnglELoss(model, scale=angle_loss_scale)

## Create Evaluator
# eval_sentences1 = [query, query, query]
# eval_sentences2 = documents
# eval_scores = [1.0, 0.0, 0.0]

# evaluator = EmbeddingSimilarityEvaluator(
#     sentences1=eval_sentences1,
#     sentences2=eval_sentences2,
#     scores=eval_scores,
#     name="sts-eval",
# )
evaluator = SampleCombinedMarginEvaluator(query=query, documents=documents, \
    q_prompt=q_prompt, d_prompt=d_prompt)

## Create Callback
# class MyCallback(TrainerCallback):
#     "A callback that evaluates the model at the end of eopch"
#     def __init__(self, evaluate):
#         self.evaluate = evaluate # evaluate function

#     def on_log(self, args, state, control, **kwargs):
#         # Evaluate the model using text generation
#         print(f"Step {state.global_step} finished. Running evaluation:")
#         self.evaluate()

# def evaluate():
#   get_scores(model, query, documents)

best_model_save_path = "my-embedding-gemma-best"
save_best_model_callback = SaveBestModelCallback(evaluator=evaluator, model=model, save_path=best_model_save_path)

## Args
# args = SentenceTransformerTrainingArguments(
#     output_dir="my-embedding-gemma",
#     num_train_epochs=30, 
#     per_device_train_batch_size=1,
#     learning_rate=2e-5,
#     warmup_ratio=0.1,
#     logging_steps=train_dataset.num_rows,
#     report_to="none",
# )

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir="my-embedding-gemma",
    # Optional training parameters:
    # prompts=model.prompts[task_name],   
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size, #1,
    learning_rate=learning_rate,
    warmup_ratio=warmup_ratio,
    # Optional tracking/debugging parameters:
    logging_steps=logging_steps, # train_dataset.num_rows,
    report_to="none",
    seed=seed,
    data_seed=seed,
)

## Initialize trainer with the correct model variable
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    # evaluator=evaluator,
    callbacks=[save_best_model_callback] #[MyCallback(evaluate)]
)

## Check Dataloader
# try:
#     # Trainer가 내부적으로 사용하는 DataLoader를 직접 가져옵니다.
#     train_dataloader = trainer.get_train_dataloader()

#     print("\n" + "="*50)
#     print("DEBUGGING INFO")
#     print("="*50)
#     # Trainer가 인식하는 데이터셋의 전체 길이
#     print(f"Trainer's dataset size (len(trainer.train_dataset)): {len(trainer.train_dataset)}")
    
#     # DataLoader가 한 에포크 동안 생성하는 배치의 총 개수 (가장 중요!)
#     print(f"Number of batches per epoch (len(train_dataloader)): {len(train_dataloader)}")
    
#     # 첫 번째 배치의 내용 확인
#     first_batch = next(iter(train_dataloader))
#     print("\n--- First Batch Content ---")
#     for key, value in first_batch.items():
#         # 텐서의 모양(shape)을 확인하여 배치 크기를 검증합니다.
#         print(f"Key: '{key}', Shape: {value.shape}")
#     print("="*50 + "\n")

# except Exception as e:
#     print(f"디버깅 중 에러 발생: {e}")

## Train
print("\n------------------ Starting Training ------------------")
# print(" - Checksum before:", model_checksum(model))
trainer.train()
print("\--------------------- Training Finished ------------------")
# print(" - Checksum after :", model_checksum(model))

## Evaluate the fine-tuned model
print('\n("\--------------------- After Training ------------------')
print(f"\n--- Loading the best model from '{best_model_save_path}' ---")
best_model = SentenceTransformer(best_model_save_path)
get_scores(best_model, query, documents, q_prompt, d_prompt)

# print(" - Checksum (in-memory, after training):", model_checksum(model))
# print(" - Checksum (loaded best model):", model_checksum(best_model))