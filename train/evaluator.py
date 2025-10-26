from sentence_transformers.evaluation import SentenceEvaluator
import torch
import numpy as np
import torch.nn.functional as F

from train.utils import add_prompt

class SampleMarginEvaluator(SentenceEvaluator):
    def __init__(self, query, documents, q_prompt=None, d_prompt=None,\
        name="margin-eval"):

        self.query = query
        self.documents = documents
        self.q_prompt = q_prompt
        self.d_prompt = d_prompt
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        
        if self.q_prompt is not None:
            query = add_prompt(self.query, self.q_prompt)
        else:
            query = self.query

        if self.d_prompt is not None:
            documents = [add_prompt(d, self.d_prompt) for d in self.documents]
        else:
            documents = self.documents

        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=False)
        # query_embedding = model.encode(query, prompt="search_query", convert_to_tensor=True, show_progress_bar=False)
        # doc_embeddings = model.encode(documents, prompt="search_document", convert_to_tensor=True, show_progress_bar=False)
        # query_embedding_nom = F.normalize(query_embedding, p=2, dim=-1)
        # doc_embeddings_nom = F.normalize(doc_embeddings, p=2, dim=-1)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
        # similarities = torch.nn.functional.cosine_similarity(query_embedding_nom, doc_embeddings_nom)

        score_nisa = similarities[0].item()
        score_regular = similarities[1].item()
        score_loan = similarities[2].item()
        
        margin = score_nisa - score_regular

        # print(f"- Query: {query}")
        # for idx, doc in enumerate(documents):
        #     # Using .item() to get a clean float number
        #     # score = similarities[0][idx].item()
        #     score = similarities[idx].item()
        #     print(f"Document: {doc} -> 🤖 Score: {score:.6f}")
        # print(f"====================================")

        print(f"\n--- Evaluator at Step {steps} ---")
        print(f"  NISA Score: {score_nisa:.6f}")
        print(f"  Regular Savings Score: {score_regular:.6f}")
        print(f"  Home Loan Score: {score_loan:.6f}")
        print(f"  => Current Margin (NISA - Regular): {margin:.6f}")
        print(f"  ------------------------------------")
        
        return margin

class SampleCombinedMarginEvaluator(SentenceEvaluator):
        
    def __init__(self, query, documents, q_prompt=None, d_prompt=None,\
            hard_margin_weight=0.7, easy_margin_weight=0.3, \
            name="combined-margin-eval"):
        self.query = query
        self.documents = documents
        self.q_prompt = q_prompt
        self.d_prompt = d_prompt
        self.name = name
        self.hard_margin_weight = hard_margin_weight
        self.easy_margin_weight = easy_margin_weight

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        
        if self.q_prompt is not None:
            query = add_prompt(self.query, self.q_prompt)
        else:
            query = self.query

        if self.d_prompt is not None:
            documents = [add_prompt(d, self.d_prompt) for d in self.documents]
        else:
            documents = self.documents

        query_embedding = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = model.encode(documents, convert_to_tensor=True, show_progress_bar=False)
        # query_embedding = model.encode(query, prompt="search_query", convert_to_tensor=True, show_progress_bar=False)
        # doc_embeddings = model.encode(documents, prompt="search_document", convert_to_tensor=True, show_progress_bar=False)
        # query_embedding_nom = F.normalize(query_embedding, p=2, dim=-1)
        # doc_embeddings_nom = F.normalize(doc_embeddings, p=2, dim=-1)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
        # similarities = torch.nn.functional.cosine_similarity(query_embedding_nom, doc_embeddings_nom)

        score_nisa = similarities[0].item()
        score_regular = similarities[1].item()
        score_loan = similarities[2].item()
        
        # 어려운 문제 Margin 계산
        margin_hard = score_nisa - score_regular
        # 쉬운 문제 Margin 계산
        margin_easy = score_nisa - score_loan
        
        # 가중치를 적용한 종합 점수 계산
        combined_score = (self.hard_margin_weight * margin_hard) + (self.easy_margin_weight * margin_easy)

        # print(f"- Query: {query}")
        # for idx, doc in enumerate(documents):
        #     # Using .item() to get a clean float number
        #     # score = similarities[0][idx].item()
        #     score = similarities[idx].item()
        #     print(f"Document: {doc} -> 🤖 Score: {score:.6f}")
        # print(f"====================================")

        print(f"\n--- Evaluator at Step {steps} ---")
        print(f"  NISA Score: {score_nisa:.6f}")
        print(f"  Regular Savings Score: {score_regular:.6f}")
        print(f"  Home Loan Score: {score_loan:.6f}")
        print(f"  ------------------------------------")
        print(f"  Hard Margin (NISA - Regular): {margin_hard:.6f} (Weight: {self.hard_margin_weight})")
        print(f"  Easy Margin (NISA - Loan): {margin_easy:.6f} (Weight: {self.easy_margin_weight})")
        print(f"  => Combined Score: {combined_score:.6f}")
        print(f"  ------------------------------------")

        return combined_score

def get_scores(model_to_eval, query, documents, q_prompt, d_prompt, task_name=None):
    # query_embeddings = model_to_eval.encode(query, prompt=task_name)
    # doc_embeddings = model_to_eval.encode(documents, prompt=task_name)
    # similarities = model_to_eval.similarity(query_embeddings, doc_embeddings)
    
    print(f"============ get_scores ============")
    model_to_eval.eval()

    query = add_prompt(query, q_prompt)
    documents = [add_prompt(d, d_prompt) for d in documents]

    # query_embedding = model_to_eval.encode(query, prompt=task_name, convert_to_tensor=True, show_progress_bar=False)
    # doc_embeddings = model_to_eval.encode(documents, prompt=task_name, convert_to_tensor=True, show_progress_bar=False)
    query_embedding = model_to_eval.encode(query, convert_to_tensor=True, show_progress_bar=False)
    doc_embeddings = model_to_eval.encode(documents, convert_to_tensor=True, show_progress_bar=False)
    # query_embedding = model_to_eval.encode(query, prompt="search_query", convert_to_tensor=True, show_progress_bar=False)
    # doc_embeddings = model_to_eval.encode(documents, prompt="search_document", convert_to_tensor=True, show_progress_bar=False)
    # query_embedding_nom = F.normalize(query_embedding, p=2, dim=-1)
    # doc_embeddings_nom = F.normalize(doc_embeddings, p=2, dim=-1)
    # similarities = torch.nn.functional.cosine_similarity(query_embedding_nom, doc_embeddings_nom)
    similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
    
    print(f"- Query: {query}")
    for idx, doc in enumerate(documents):
        # Using .item() to get a clean float number
        # score = similarities[0][idx].item()
        score = similarities[idx].item()
        print(f"Document: {doc} -> 🤖 Score: {score:.6f}")
    print(f"====================================")