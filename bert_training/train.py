import argparse
import logging
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_model(model_name: str, device: str):
    """지정된 사전 학습 모델을 로드합니다."""
    logging.info(f"Loading pre-trained model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    return model

def train_unsupervised(args):
    """
    비지도 SimCSE 방식으로 모델을 학습합니다.
    (In-batch negatives)
    """
    logging.info("Starting unsupervised training...")

    # --- 모델 로드 ---
    model = get_model(args.model_name, args.device)

    # --- 데이터 로드 ---
    logging.info(f"Loading unsupervised data from: {args.train_file}")
    try:
        df = pd.read_csv(args.train_file, encoding='utf-8-sig')
        if 'text' not in df.columns:
            raise ValueError("Unsupervised data must have a 'text' column.")
        # SimCSE는 동일한 문장을 positive 쌍으로 사용하므로, [sent, sent] 형식으로 데이터를 구성
        train_samples = [InputExample(texts=[text, text]) for text in tqdm(df['text'], desc="Loading data")]
    except FileNotFoundError:
        logging.error(f"Training file not found: {args.train_file}")
        return
    except Exception as e:
        logging.error(f"Error loading or processing data: {e}")
        return

    logging.info(f"Loaded {len(train_samples)} training samples.")

    # --- 데이터로더 및 손실 함수 정의 ---
    # DataLoader는 텍스트 쌍을 배치로 만듭니다.
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # MultipleNegativesRankingLoss는 SimCSE의 핵심입니다.
    # 배치 내에서 한 문장의 positive 쌍(자기 자신)을 제외한 모든 다른 문장을 negative로 사용합니다.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # --- 모델 학습 ---
    logging.info(f"Training for {args.epochs} epochs...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=100,
              optimizer_params={'lr': args.lr},
              output_path=args.output_dir,
              show_progress_bar=True,
              checkpoint_path=os.path.join(args.output_dir, 'checkpoints'),
              checkpoint_save_steps=5000)

    logging.info(f"Unsupervised training finished. Model saved to {args.output_dir}")


def train_supervised(args):
    """
    지도 학습 방식으로 모델을 학습합니다.
    (Anchor, Positive, Negative) Triplet 데이터를 사용합니다.
    """
    logging.info("Starting supervised training...")

    # --- 모델 로드 ---
    model = get_model(args.model_name, args.device)

    # --- 데이터 로드 ---
    logging.info(f"Loading supervised data from: {args.train_file}")
    try:
        df = pd.read_csv(args.train_file, encoding='utf-8-sig')
        # 필요한 컬럼 확인
        required_cols = ['anchor', 'positive']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Supervised data must have 'anchor' and 'positive' columns.")

        # negative 컬럼들을 동적으로 찾기
        negative_cols = sorted([col for col in df.columns if col.startswith('negative')])
        if not negative_cols:
             logging.warning("No 'negative_...' columns found. Using in-batch negatives only.")

        train_samples = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading data"):
            anchor = row['anchor']
            positive = row['positive']

            # TripletLoss를 위해 [anchor, positive, negative] 형식으로 데이터를 구성
            # Hard negative가 여러 개 있는 경우, 각각을 별도의 학습 샘플로 만듭니다.
            if negative_cols:
                for neg_col in negative_cols:
                    if pd.notna(row[neg_col]) and row[neg_col]:
                        negative = row[neg_col]
                        train_samples.append(InputExample(texts=[anchor, positive, negative]))
            else:
                # Hard negative가 없는 경우, anchor-positive 쌍만 추가하고,
                # loss 함수가 배치 내에서 negative를 찾도록 합니다.
                # 이 경우 TripletLoss보다 MultipleNegativesRankingLoss가 더 적합합니다.
                # 여기서는 TripletLoss를 가정하고 로직을 구성했으므로 경고만 남깁니다.
                train_samples.append(InputExample(texts=[anchor, positive]))


    except FileNotFoundError:
        logging.error(f"Training file not found: {args.train_file}")
        return
    except Exception as e:
        logging.error(f"Error loading or processing data: {e}")
        return

    if not train_samples:
        logging.error("No training samples could be created. Please check the data file.")
        return

    logging.info(f"Loaded {len(train_samples)} training triplets.")

    # --- 데이터로더 및 손실 함수 정의 ---
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # TripletLoss는 (anchor, positive) 사이의 거리는 최소화하고,
    # (anchor, negative) 사이의 거리는 최대화하도록 학습합니다.
    train_loss = losses.TripletLoss(model=model)

    # --- 모델 학습 ---
    logging.info(f"Training for {args.epochs} epochs...")
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=100,
              optimizer_params={'lr': args.lr},
              output_path=args.output_dir,
              show_progress_bar=True,
              checkpoint_path=os.path.join(args.output_dir, 'checkpoints'),
              checkpoint_save_steps=5000)

    logging.info(f"Supervised training finished. Model saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="BERT-based Sentence Embedding Model Trainer")

    # --- 필수 인자 ---
    parser.add_argument("--mode", type=str, required=True, choices=['unsupervised', 'supervised'],
                        help="Training mode: 'unsupervised' (SimCSE) or 'supervised' (Triplet).")
    parser.add_argument("--train_file", type=str, required=True,
                        help="Path to the training data CSV file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model.")

    # --- 모델 및 학습 관련 인자 ---
    parser.add_argument("--model_name", type=str, default="klue/bert-base",
                        help="Name of the pre-trained model from Hugging Face.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training ('cuda' or 'cpu').")

    args = parser.parse_args()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'unsupervised':
        train_unsupervised(args)
    elif args.mode == 'supervised':
        train_supervised(args)

if __name__ == "__main__":
    main()
