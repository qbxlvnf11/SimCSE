import pandas as pd
import random
import argparse
import sys
import os

def process_dataframe(df, desc_col, category_col, positive_cols):

    # 어려운 부정 쌍(Hard Negative)을 쉽게 찾기 위해 '분류'별로 데이터 그룹화
    category_groups = df.groupby(category_col)[desc_col].apply(list).to_dict()

    unsupervised_samples = []
    supervised_triplets = []
    
    # --- 비지도 학습에 사용될 컬럼 리스트 생성 ---
    all_text_cols = [desc_col] + positive_cols

    for index, row in df.iterrows():
        description = str(row[desc_col]).strip()
        if not description:
            continue

        # --- 1. 비지도 학습용 데이터 생성 ---
        parts = []
        for col_name in sorted(all_text_cols):
             parts.append(f"[{col_name}] {str(row[col_name]).strip()}")
        templated_sentence = " ".join(parts)
        unsupervised_samples.append(templated_sentence)

        # --- 2. 지도 학습용 Triplet 데이터 생성 ---
        anchor = description
        
        current_positive_cols = [category_col] + positive_cols
        for pos_col_name in current_positive_cols:
            positive_text = str(row[pos_col_name]).strip()
            if not positive_text or positive_text == anchor:
                continue

            # 어려운 부정 쌍 찾기
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
    parser.add_argument("--positive_cols", type=str, nargs='+', default=["표제어", "영문"],
                        help="긍정 쌍(Positive)으로 사용할 컬럼명 리스트 (스페이스로 구분)")
    parser.add_argument("--encoding", type=str, default="utf-8")

    parser.add_argument("--domain_col", type=str, required=True, help="(선택) 데이터를 분리할 기준이 되는 도메인 컬럼명")
    parser.add_argument("--output_dir", type=str, help="(선택) 결과 파일을 저장할 디렉터리 경로")
    
    args = parser.parse_args()

    # --- 데이터 읽기 ---
    try:
        df = pd.read_csv(args.data_path, encoding=args.encoding, engine='python').fillna('')
        print(f"✅ 총 {len(df)}개의 데이터를 성공적으로 읽었습니다.")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {args.data_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ 파일 읽기 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 필요한 컬럼 확인 ---
    required_cols = [args.desc_col, args.category_col] + args.positive_cols
    if args.domain_col:
        required_cols.append(args.domain_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 파일에 필요한 컬럼이 없습니다: {', '.join(missing_cols)}", file=sys.stderr)
        sys.exit(1)

    # --- 도메인별 처리 또는 전체 처리 ---
    domain_groups = {}
    if args.domain_col:
        # 도메인 컬럼의 고유값으로 데이터프레임을 그룹화
        for domain_name, domain_df in df.groupby(args.domain_col):
            domain_groups[str(domain_name)] = domain_df
    else:
        # 도메인 컬럼이 없으면 'all'이라는 이름으로 전체 데이터프레임을 처리
        domain_groups['all'] = df

    for domain, data_df in domain_groups.items():
        print(f"\n===== [ 도메인: {domain} ] 처리 시작 (데이터 {len(data_df)}개) =====")
        
        # --- 데이터 생성 함수 호출 ---
        unsupervised_data, supervised_data = process_dataframe(
            df=data_df,
            desc_col=args.desc_col,
            category_col=args.category_col,
            positive_cols=args.positive_cols,
        )

        # --- 파일 저장 로직 ---
        if args.output_dir:
            # 출력 디렉터리가 없으면 생성
            os.makedirs(args.output_dir, exist_ok=True)
            domain = domain.replace('/', '_')
            
            # 비지도 학습 데이터 저장
            if unsupervised_data:
                unsupervised_filename = os.path.join(args.output_dir, f"{domain}_unsupervised.csv")
                df_unsupervised = pd.DataFrame(unsupervised_data, columns=['text'])
                df_unsupervised.to_csv(unsupervised_filename, index=False, encoding='utf-8-sig')
                print(f"💾 비지도 학습 데이터셋을 '{unsupervised_filename}'에 저장했습니다.")

            # 지도 학습 데이터 저장
            if supervised_data:
                supervised_filename = os.path.join(args.output_dir, f"{domain}_supervised.csv")
                df_supervised = pd.DataFrame(supervised_data)
                df_supervised.to_csv(supervised_filename, index=False, encoding='utf-8-sig')
                print(f"💾 지도 학습 데이터셋을 '{supervised_filename}'에 저장했습니다.")
        else:
             # 출력 인자가 없을 경우, 샘플 데이터만 화면에 출력
            print("\n--- [비지도 학습 샘플 (상위 3개)] ---")
            for i, sentence in enumerate(unsupervised_data[:3]):
                print(f"{i+1}: {sentence}")

            print("\n--- [지도 학습 Triplet 샘플 (상위 3개)] ---")
            for i, triplet in enumerate(supervised_data[:3]):
                print(f"{i+1}: {triplet}")


if __name__ == "__main__":
    main()
