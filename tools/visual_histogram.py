# import numpy as np
# import matplotlib.pyplot as plt
import os

# def plot_histograms_from_npz(npz_path, save_dir):
#     data = np.load(npz_path)
#     similarities = data['similarities']
#     labels = data['labels']
#     glasses_pairs = data['glasses_pairs']
    
#     min_sim = similarities.min()
#     max_sim = similarities.max()
#     xlim = (min_sim, max_sim)

#     # ==== [디버깅 코드 시작] ====
#     print("="*30)
#     print(f"Total samples: {len(labels)}")
#     print(f"Unique labels: {np.unique(labels, return_counts=True)}")
#     print(f"genuine 개수: {np.sum(labels==1)}, impostor 개수: {np.sum(labels==0)}")
#     print(f"Unique glasses_pairs: {np.unique(glasses_pairs, return_counts=True)}")
#     print(f"similarities: min={min_sim}, max={max_sim}, mean={similarities.mean():.2f}, std={similarities.std():.2f}")

#     # 각 마스크 적용 후 개수
#     print("--- Mask counts ---")
#     print("genuine_all:", np.sum(labels==1))
#     print("impostor_all:", np.sum(labels==0))
#     print("overall_all:", np.sum(np.ones_like(labels, dtype=bool)))


#     def save_histogram(title, filename, masks, xlim=None):
#         plt.figure(figsize=(8, 6))
#         for mask, label_name in masks:
#             plt.hist(similarities[mask], bins=50, alpha=0.5, label=label_name, density=True)
#         plt.xlabel("Raw Similarity Score (before sigmoid)")
#         plt.ylabel("Density")
#         plt.title(title)
#         plt.legend()
#         if xlim is not None:
#             plt.xlim(xlim)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, filename))
#         plt.close()
        
#     os.makedirs(save_dir, exist_ok=True)

#     # 1. Pair별 Genuine/Impostor
#     for pair_val in [0, 1, 2]:
#         mask_genuine = (labels == 1) & (glasses_pairs == pair_val)
#         mask_impostor = (labels == 0) & (glasses_pairs == pair_val)
#         save_histogram(
#             f"Similarity Distribution (Pair {pair_val})",
#             f"hist_pair{pair_val}.png",
#             [(mask_genuine, "Genuine"), (mask_impostor, "Impostor")],
#             xlim=xlim
#         )

#     # 2. 전체 Genuine/Impostor - pair별 레이오버
#     save_histogram(
#         "Genuine Distribution Comparison",
#         "genuine_compare.png",
#         [((labels == 1) & (glasses_pairs == val), f"Pair {val}") for val in [0, 1, 2]],
#         xlim=xlim
#     )

#     save_histogram(
#         "Impostor Distribution Comparison",
#         "impostor_compare.png",
#         [((labels == 0) & (glasses_pairs == val), f"Pair {val}") for val in [0, 1, 2]],
#         xlim=xlim
#     )

#     # 3. 전체(모든 쌍) - pair별 레이오버
#     save_histogram(
#         "Overall Score Distribution",
#         "overall_compare.png",
#         [(glasses_pairs == val, f"Pair {val}") for val in [0, 1, 2]],
#         xlim=xlim
#     )

#     # 4. 전체 Genuine only, Impostor only, All (pair 구분 없이 통합)
#     save_histogram(
#         "Genuine Overall Similarity Distribution",
#         "genuine_all.png",
#         [((labels == 1), "Genuine (All)")],
#         xlim=xlim
#     )
#     save_histogram(
#         "Impostor Overall Similarity Distribution",
#         "impostor_all.png",
#         [((labels == 0), "Impostor (All)")],
#         xlim=xlim
#     )
#     save_histogram(
#         "Overall Similarity Distribution",
#         "overall_all.png",
#         [(np.ones_like(labels, dtype=bool), "All (Genuine+Impostor)")],
#         xlim=xlim
#     )

# # 사용 예시 (실행 방법)
# plot_histograms_from_npz(
#     './results/original_vanilla_fold2_test/epoch_100/evaldata_epoch_100.npz',
#     './results/original_vanilla_fold2_test/epoch_100/visualization'
# # )
# import pandas as pd

# # 파일 경로에 맞게 수정
# file_path = './periocular_data/fold1_pairs_train.txt'  # 예: 'data.txt'

# # 탭 또는 공백으로 구분되어 있다고 가정
# df = pd.read_csv(file_path, sep='\t|\s+', header=None, engine='python')

# # 4번째 컬럼의 값 분포 확인 (인덱스 3)
# value_counts = df[2].value_counts()
# # print("4번째 컬럼 분포:")
# print(value_counts)

import os
from collections import Counter, defaultdict

def analyze_pair_file(pair_file):
    genuine_cnt = 0
    impostor_cnt = 0
    domain_counter = Counter()
    pair_counter = defaultdict(list)  # (is_genuine, domain): 리스트

    with open(pair_file, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) != 4:
                continue
            _, _, is_genuine, domain = fields
            is_genuine = int(is_genuine)
            domain = int(domain)
            # 전체 도메인 카운트
            domain_counter[domain] += 1
            pair_counter[(is_genuine, domain)].append(fields)
            # Genuine / Impostor 카운트
            if is_genuine == 1:
                genuine_cnt += 1
            else:
                impostor_cnt += 1

    print(f"\n[{os.path.basename(pair_file)}] 통계")
    print(f"전체 쌍 수: {genuine_cnt + impostor_cnt}")
    print(f"Genuine 쌍: {genuine_cnt}")
    print(f"Impostor 쌍: {impostor_cnt}")
    if impostor_cnt > 0:
        print(f"Genuine:Impostor 비율 = {genuine_cnt}:{impostor_cnt} ({genuine_cnt/impostor_cnt:.2f}:1)")
    else:
        print("Impostor 쌍 없음")
    print("\n도메인별 쌍 개수:")
    for d in range(3):
        dom_g = len(pair_counter[(1, d)])
        dom_i = len(pair_counter[(0, d)])
        print(f"  Domain {d} - Genuine: {dom_g}, Impostor: {dom_i}, Total: {domain_counter[d]}")
    print("-" * 40)

# 사용 예시
if __name__ == "__main__":
    root_dir = "C:\\Users\\SELIM\\Desktop\\Periocular_Biometric-glasses-lee\\sampling_data_origin_total"  # 페어 파일이 저장된 경로
    for fname in os.listdir(root_dir):
        if fname.endswith(".txt") and "pairs" in fname:
            pair_path = os.path.join(root_dir, fname)
            analyze_pair_file(pair_path)
