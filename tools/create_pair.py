import os
import itertools
import random
from tqdm import tqdm

def get_person_images(root_dir, person):
    """
    한 피험자 폴더에서 안경/비안경 이미지 리스트 반환
    """
    glasses_imgs, no_glasses_imgs = [], []
    root_dir_name = os.path.basename(os.path.normpath(root_dir))
    person_path = os.path.join(root_dir, person)
    no_glass_path = os.path.join(person_path, "no_glass")
    glass_path = os.path.join(person_path, "glass")

    if os.path.isdir(no_glass_path):
        no_glasses_imgs = [
            f"./{root_dir_name}/{person}/no_glass/{frame}"
            for frame in os.listdir(no_glass_path)
            if frame.lower().endswith('.jpg')
        ]
    if os.path.isdir(glass_path):
        glasses_imgs = [
            f"./{root_dir_name}/{person}/glass/{frame}"
            for frame in os.listdir(glass_path)
            if frame.lower().endswith('.jpg')
        ]
    return glasses_imgs, no_glasses_imgs

def make_pairs(persons, root_dir):
    """
    지정된 피험자 집합으로 genuine/impostor 쌍 생성 (도메인별 균등)
    """
    # 1. 피험자별 이미지 읽기 및 최소 개수 맞추기
    person_images = {}
    min_img_per_person = None
    for person in persons:
        glasses_imgs, no_glasses_imgs = get_person_images(root_dir, person)
        img_count = min(len(glasses_imgs), len(no_glasses_imgs))
        if img_count == 0:
            continue
        if min_img_per_person is None or img_count < min_img_per_person:
            min_img_per_person = img_count
        # 임시저장
        person_images[person] = {
            "glasses": glasses_imgs,
            "no_glasses": no_glasses_imgs
        }
    # 최소 개수만큼 샘플링
    for person in person_images:
        person_images[person]["no_glasses"] = random.sample(
            person_images[person]["no_glasses"], min_img_per_person)
        person_images[person]["glasses"] = random.sample(
            person_images[person]["glasses"], min_img_per_person)
    person_list = list(person_images.keys())

    # 2. genuine 쌍 생성
    genuine_pairs_0, genuine_pairs_1, genuine_pairs_2 = [], [], []
    for person in person_images:
        images = person_images[person]
        glasses_imgs = images["glasses"]
        no_glasses_imgs = images["no_glasses"]
        # domain 0: no_glass-no_glass
        for img1, img2 in itertools.combinations(no_glasses_imgs, 2):
            genuine_pairs_0.append((img1, img2, 1, 0))
        # domain 1: glass-glass
        for img1, img2 in itertools.combinations(glasses_imgs, 2):
            genuine_pairs_1.append((img1, img2, 1, 1))
        # domain 2: cross
        for img1 in no_glasses_imgs:
            for img2 in glasses_imgs:
                genuine_pairs_2.append((img1, img2, 1, 2))
    # 균등 샘플링
    min_len = min(len(genuine_pairs_0), len(genuine_pairs_1), len(genuine_pairs_2))
    random.shuffle(genuine_pairs_0)
    random.shuffle(genuine_pairs_1)
    random.shuffle(genuine_pairs_2)
    genuine_pairs_0 = genuine_pairs_0[:min_len]
    genuine_pairs_1 = genuine_pairs_1[:min_len]
    genuine_pairs_2 = genuine_pairs_2[:min_len]
    genuine_pairs = genuine_pairs_0 + genuine_pairs_1 + genuine_pairs_2
    random.shuffle(genuine_pairs)

    # 3. impostor 쌍 생성 (도메인별 균등)
    impostor_types = {0: [], 1: [], 2: []}
    target_per_domain = len(genuine_pairs)
    while sum(len(impostor_types[k]) for k in impostor_types) < target_per_domain:
        p1, p2 = random.sample(person_list, 2)
        # domain 0: no_glass-no_glass
        if len(impostor_types[0]) < target_per_domain // 3 and person_images[p1]["no_glasses"] and person_images[p2]["no_glasses"]:
            img1 = random.choice(person_images[p1]["no_glasses"])
            img2 = random.choice(person_images[p2]["no_glasses"])
            impostor_types[0].append((img1, img2, 0, 0))
        # domain 1: glass-glass
        if len(impostor_types[1]) < target_per_domain // 3 and person_images[p1]["glasses"] and person_images[p2]["glasses"]:
            img1 = random.choice(person_images[p1]["glasses"])
            img2 = random.choice(person_images[p2]["glasses"])
            impostor_types[1].append((img1, img2, 0, 1))
        # domain 2: cross
        if len(impostor_types[2]) < target_per_domain // 3:
            if person_images[p1]["no_glasses"] and person_images[p2]["glasses"]:
                img1 = random.choice(person_images[p1]["no_glasses"])
                img2 = random.choice(person_images[p2]["glasses"])
                impostor_types[2].append((img1, img2, 0, 2))
            elif person_images[p1]["glasses"] and person_images[p2]["no_glasses"]:
                img1 = random.choice(person_images[p1]["glasses"])
                img2 = random.choice(person_images[p2]["no_glasses"])
                impostor_types[2].append((img1, img2, 0, 2))
    impostor_pairs = impostor_types[0] + impostor_types[1] + impostor_types[2]
    random.shuffle(impostor_pairs)
    pairs = genuine_pairs + impostor_pairs
    random.shuffle(pairs)

    return pairs, person_list

if __name__ == "__main__":
    root_dir = "./periocular_data"  # 데이터가 있는 디렉토리 경로

    # 1. 전체 피험자 리스트 준비
    all_subjects = [person for person in os.listdir(root_dir)
                    if os.path.isdir(os.path.join(root_dir, person))]

    # 2. Fold 나누기
    random.seed(2024)
    subjects = all_subjects[:]
    random.shuffle(subjects)
    n = len(subjects)
    fold_sizes = [n // 5] * 5
    for i in range(n % 5):
        fold_sizes[i] += 1
    folds = []
    idx = 0
    for size in fold_sizes:
        folds.append(subjects[idx:idx+size])
        idx += size

    print("\n===== Fold 피험자 그룹 구성 =====")
    for i, fold in enumerate(folds, 1):
        print(f"Fold {i} ({len(fold)}명): {fold}")

    # 3. Fold별로 쌍 생성 및 저장
    for i, valid_subjects in enumerate(folds, 1):
        # valid set: 이 fold의 피험자
        # train set: 나머지 피험자
        train_subjects = []
        for j, fold in enumerate(folds):
            if i-1 != j:
                train_subjects.extend(fold)
        # 쌍 생성
        train_pairs, train_persons = make_pairs(train_subjects, root_dir)
        valid_pairs, valid_persons = make_pairs(valid_subjects, root_dir)
        # 저장
        train_file = os.path.join(root_dir, f"fold{i}_pairs_train.txt")
        valid_file = os.path.join(root_dir, f"fold{i}_pairs_valid.txt")
        subj_file  = os.path.join(root_dir, f"fold{i}_subjects.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            for pair in train_pairs:
                f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
        with open(valid_file, "w", encoding="utf-8") as f:
            for pair in valid_pairs:
                f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")
        with open(subj_file, "w", encoding="utf-8") as f:
            for subj in valid_subjects:
                f.write(subj + '\n')
        print(f"\nFold {i} 결과: train={len(train_pairs)}, valid={len(valid_pairs)}")

