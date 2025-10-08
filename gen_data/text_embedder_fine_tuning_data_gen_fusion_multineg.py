import pandas as pd
import random
import argparse
import sys
import os

def process_dataframe(df, desc_col, category_col, positive_cols, num_negatives=5):

    category_groups = df.groupby(category_col)[desc_col].apply(list).to_dict()

    unsupervised_samples = []
    supervised_samples = []
    
    all_text_cols = [desc_col] + positive_cols

    for index, row in df.iterrows():
        description = str(row[desc_col]).strip()
        if not description:
            continue

        # 1. 비지도 학습용 데이터 생성
        parts = []
        for col_name in sorted(all_text_cols):
             parts.append(f"[{col_name}] {str(row[col_name]).strip()}")
        templated_sentence = " ".join(parts)
        unsupervised_samples.append(templated_sentence)

        # --- 2. 지도 학습용 데이터 생성 (변경 2 로직 적용) ---
        anchor = description
        
        # Positive 데이터 통합
        positive_parts = []
        for pos_col in positive_cols:
            col_value = str(row[pos_col]).strip()
            if col_value: # 값이 있는 경우에만 추가
                positive_parts.append(f"{pos_col}: {col_value}")
        positive_combined = ", ".join(positive_parts)

        if not positive_combined:
            continue

        # 어려운 부정 쌍(Hard Negative) 후보군 탐색
        category = str(row[category_col]).strip()
        hard_negative_candidates = [
            desc for desc in category_groups.get(category, []) if desc != anchor
        ]

        # 지정된 개수만큼 Hard Negative 샘플링
        if hard_negative_candidates:
            # 후보군이 부족할 경우 중복 샘플링(choices) 허용
            negatives = random.choices(hard_negative_candidates, k=num_negatives)

            new_row = {
                'anchor': anchor,
                'positive': positive_combined,
            }
            # negative_1, negative_2, ... 컬럼 추가
            for i, neg in enumerate(negatives):
                new_row[f'negative_{i+1}'] = neg
            
            supervised_samples.append(new_row)

    print(f"✅ 비지도 학습 샘플 {len(unsupervised_samples)}개 생성 완료.")
    print(f"✅ 지도 학습 샘플 {len(supervised_samples)}개 생성 완료.")
    
    return unsupervised_samples, supervised_samples

def main():
    parser = argparse.ArgumentParser(description="SimCSE 학습을 위한 데이터셋 생성 스크립트 (변경 2 적용)")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="입력할 원본 CSV 파일 경로")
    parser.add_argument("--desc_col", type=str, default="설명",
                        help="기준 문장(Anchor)으로 사용할 설명 컬럼명")
    parser.add_argument("--category_col", type=str, default="대분류",
                        help="그룹화 및 Hard Negative 탐색에 사용할 분류 컬럼명")
    parser.add_argument("--positive_cols", type=str, nargs='+', 
                        default=["대분류", "중분류", "표제어", "영문"],
                        help="긍정 쌍(Positive)으로 사용할 컬럼명 리스트 (스페이스로 구분)")
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument("--domain_col", type=str, help="(선택) 데이터를 분리할 기준이 되는 도메인 컬럼명")
    parser.add_argument("--output_dir", type=str, help="(선택) 결과 파일을 저장할 디렉터리 경로")
    parser.add_argument("--num_negatives", type=int, default=5,
                        help="생성할 Hard Negative 샘플의 개수")
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.data_path, encoding=args.encoding, engine='python').fillna('')
        print(f"✅ 총 {len(df)}개의 데이터를 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {args.data_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 파일 읽기 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    required_cols = [args.desc_col, args.category_col] + args.positive_cols
    if args.domain_col:
        required_cols.append(args.domain_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 파일에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    domain_groups = {}
    if args.domain_col:
        for domain_name, domain_df in df.groupby(args.domain_col):
            # 파일명으로 사용하기 어려운 '/' 문자를 '_'로 변경
            safe_domain_name = str(domain_name).replace('/', '_')
            domain_groups[safe_domain_name] = domain_df
    else:
        domain_groups['all'] = df

    for domain, data_df in domain_groups.items():
        print(f"\n===== [ 도메인: {domain} ] 처리 시작 (데이터 {len(data_df)}개) =====")
        
        unsupervised_data, supervised_data = process_dataframe(
            df=data_df,
            desc_col=args.desc_col,
            category_col=args.category_col,
            positive_cols=args.positive_cols,
            num_negatives=args.num_negatives
        )

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
            if unsupervised_data:
                unsupervised_filename = os.path.join(args.output_dir, f"{domain}_unsupervised.csv")
                df_unsupervised = pd.DataFrame(unsupervised_data, columns=['text'])
                df_unsupervised.to_csv(unsupervised_filename, index=False, encoding='utf-8-sig')
                print(f"💾 비지도 학습 데이터셋을 '{unsupervised_filename}'에 저장했습니다.")

            if supervised_data:
                supervised_filename = os.path.join(args.output_dir, f"{domain}_supervised.csv")
                df_supervised = pd.DataFrame(supervised_data)
                df_supervised.to_csv(supervised_filename, index=False, encoding='utf-8-sig')
                print(f"💾 지도 학습 데이터셋을 '{supervised_filename}'에 저장했습니다.")
        else:
            print("\n--- [비지도 학습 샘플 (상위 3개)] ---")
            for i, sentence in enumerate(unsupervised_data[:3]):
                print(f"{i+1}: {sentence}")

            print("\n--- [지도 학습 샘플 (상위 3개)] ---")
            for i, sample in enumerate(supervised_data[:3]):
                print(f"{i+1}: {sample}")

if __name__ == "__main__":
    main()
