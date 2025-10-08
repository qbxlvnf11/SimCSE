from sentence_transformers.evaluation import SentenceEvaluator
import torch
import numpy as np

class SampleMarginEvaluator(SentenceEvaluator):
    def __init__(self, query, documents, name="margin-eval"):
        self.query = query
        self.documents = documents
        self.name = name

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        query_embedding = model.encode(self.query, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = model.encode(self.documents, convert_to_tensor=True, show_progress_bar=False)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
        
        score_nisa = similarities[0].item()
        score_regular = similarities[1].item()
        score_loan = similarities[2].item()
        
        margin = score_nisa - score_regular

        print(f"\n--- Evaluator at Step {steps} ---")
        print(f"  NISA Score: {score_nisa:.6f}")
        print(f"  Regular Savings Score: {score_regular:.6f}")
        print(f"  Home Loan Score: {score_loan:.6f}")
        print(f"  => Current Margin (NISA - Regular): {margin:.6f}")
        
        return margin

class SampleCombinedMarginEvaluator(SentenceEvaluator):
        
    def __init__(self, query, documents, \
            hard_margin_weight=0.7, easy_margin_weight=0.3, \
            name="combined-margin-eval"):
        self.query = query
        self.documents = documents
        self.name = name
        self.hard_margin_weight = hard_margin_weight
        self.easy_margin_weight = easy_margin_weight

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        query_embedding = model.encode(self.query, convert_to_tensor=True, show_progress_bar=False)
        doc_embeddings = model.encode(self.documents, convert_to_tensor=True, show_progress_bar=False)
        similarities = torch.nn.functional.cosine_similarity(query_embedding, doc_embeddings)
        
        score_nisa = similarities[0].item()
        score_regular = similarities[1].item()
        score_loan = similarities[2].item()
        
        # 어려운 문제 Margin 계산
        margin_hard = score_nisa - score_regular
        # 쉬운 문제 Margin 계산
        margin_easy = score_nisa - score_loan
        
        # 가중치를 적용한 종합 점수 계산
        combined_score = (self.hard_margin_weight * margin_hard) + (self.easy_margin_weight * margin_easy)

        print(f"\n--- Evaluator at Step {steps} ---")
        print(f"  NISA Score: {score_nisa:.6f}")
        print(f"  Regular Savings Score: {score_regular:.6f}")
        print(f"  Home Loan Score: {score_loan:.6f}")
        print(f"  ------------------------------------")
        print(f"  Hard Margin (NISA - Regular): {margin_hard:.6f} (Weight: {self.hard_margin_weight})")
        print(f"  Easy Margin (NISA - Loan): {margin_easy:.6f} (Weight: {self.easy_margin_weight})")
        print(f"  => Combined Score: {combined_score:.6f}")
        
        return combined_score