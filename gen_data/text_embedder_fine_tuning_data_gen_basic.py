import pandas as pd
import random
import argparse
import sys

def create_training_data(file_path, desc_col, category_col, positive_cols, encoding='utf-8'):
    """
    CSV 파일을 읽어 SimCSE 하이브리드 학습용 데이터셋을 생성합니다.
    argparse를 통해 파일 경로와 컬럼명을 유연하게 지정할 수 있습니다.

    Args:
        file_path (str): 입력 CSV 파일 경로.
        desc_col (str): 기준 문장(Anchor)으로 사용할 설명 컬럼명.
        category_col (str): 데이터를 그룹화할 분류 컬럼명.
        positive_cols (list): Positive로 사용할 컬럼명 리스트.

    Returns:
        tuple: (unsupervised_samples, supervised_triplets) 튜플을 반환합니다.
    """

    print(f"encoding: {encoding}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding, engine='python').fillna('')
        print(f"✅ 총 {len(df)}개의 데이터를 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 파일 읽기 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    # 필요한 모든 컬럼이 존재하는지 확인
    required_cols = [desc_col] + positive_cols
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 파일에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    # Hard Negative을 쉽게 찾기 위해 '분류'별로 데이터 그룹화
    category_groups = df.groupby(category_col)[desc_col].apply(list).to_dict()

    unsupervised_samples = []
    supervised_triplets = []

    all_text_cols = {col: col for col in required_cols}

    for index, row in df.iterrows():
        description = str(row[desc_col]).strip()
        if not description:
            continue

        # --- 1. 비지도 학습용 데이터 생성 ---
        parts = []
        for col_name in sorted(all_text_cols.keys()):
             parts.append(f"[{col_name}] {str(row[col_name]).strip()}")
        templated_sentence = " ".join(parts)
        unsupervised_samples.append(templated_sentence)

        # --- 2. 지도 학습용 Triplet 데이터 생성 ---
        anchor = description
        
        # Positive Pairs 후보군: category 컬럼과 positive_cols에 지정된 컬럼들
        current_positive_cols = [category_col] + positive_cols
        for pos_col_name in current_positive_cols:
            positive_text = str(row[pos_col_name]).strip()
            if not positive_text or positive_text == anchor:
                continue

            # Hard Negative Pairs 찾기
            category = str(row[category_col]).strip()
            hard_negative_candidates = [
                desc for desc in category_groups.get(category, []) if desc != anchor
            ]

            if hard_negative_candidates:
                hard_negative = random.choice(hard_negative_candidates)
                supervised_triplets.append({
                    'anchor': anchor,
                    'positive': positive_text,
                    'hard_negative': hard_negative
                })

    print(f"✅ 비지도 학습 샘플 {len(unsupervised_samples)}개 생성 완료.")
    print(f"✅ 지도 학습 Triplet 샘플 {len(supervised_triplets)}개 생성 완료.")
    
    return unsupervised_samples, supervised_triplets

def main():
    # --- Argparse 설정 ---
    parser = argparse.ArgumentParser(description="SimCSE 학습을 위한 데이터셋 생성 스크립트")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="입력할 원본 CSV 파일 경로")
    parser.add_argument("--desc_col", type=str, default="설명",
                        help="기준 문장(Anchor)으로 사용할 설명 컬럼명")
    parser.add_argument("--category_col", type=str, default="분류",
                        help="그룹화 및 Hard Negative 탐색에 사용할 분류 컬럼명")
    parser.add_argument("--positive_cols", type=str, nargs='+', default=["대분류", "중분류"],
                        help="Positive으로 사용할 컬럼명 리스트 (스페이스로 구분)")
    parser.add_argument("--encoding", type=str, default="utf-8")

    parser.add_argument("--output_unsupervised", type=str, help="(선택) 비지도 학습 데이터를 저장할 CSV 파일 경로")
    parser.add_argument("--output_supervised", type=str, help="(선택) 지도 학습 데이터를 저장할 CSV 파일 경로")
    
    args = parser.parse_args()

    # --- 데이터 생성 함수 호출 ---
    unsupervised_data, supervised_data = create_training_data(
        file_path=args.data_path,
        desc_col=args.desc_col,
        category_col=args.category_col,
        positive_cols=args.positive_cols,
        encoding=args.encoding
    )

    # --- 결과 샘플 출력 ---
    print("\n--- [비지도 학습 샘플 (상위 3개)] ---")
    for i, sentence in enumerate(unsupervised_data[:3]):
        print(f"{i+1}: {sentence}")

    print("\n--- [지도 학습 Triplet 샘플 (상위 3개)] ---")
    for i, triplet in enumerate(supervised_data[:3]):
        print(f"{i+1}:")
        print(f"  - Anchor: {triplet['anchor']}")
        print(f"  - Positive: {triplet['positive']}")
        print(f"  - Hard Negative: {triplet['hard_negative']}")

    if args.output_unsupervised:
        df_unsupervised = pd.DataFrame(unsupervised_data, columns=['text'])
        df_unsupervised.to_csv(args.output_unsupervised, index=False, encoding='utf-8-sig')
        print(f"💾 비지도 학습 데이터셋을 '{args.output_unsupervised}'에 저장했습니다.")

    if args.output_supervised:
        df_supervised = pd.DataFrame(supervised_data)
        df_supervised.to_csv(args.output_supervised, index=False, encoding='utf-8-sig')
        print(f"💾 지도 학습 데이터셋을 '{args.output_supervised}'에 저장했습니다.")

    # 출력 인자가 없을 경우, 샘플 데이터만 화면에 출력
    if not args.output_unsupervised and not args.output_supervised:
        print("\n--- [비지도 학습 샘플 (상위 3개)] ---")
        for i, sentence in enumerate(unsupervised_data[:3]):
            print(f"{i+1}: {sentence}")

        print("\n--- [지도 학습 Triplet 샘플 (상위 3개)] ---")
        for i, triplet in enumerate(supervised_data[:3]):
            print(f"{i+1}: {triplet}")

if __name__ == "__main__":

    main()
