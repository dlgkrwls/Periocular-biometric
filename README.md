# RockSiN: 안경 착용 변화에 강인한 Periocular Cross-domain Biometrics

> **📌 현재 논문 리뷰 중이며, 전체 코드 및 모델 가중치는 Accept 이후 순차적으로 공개될 예정입니다.**  
> 본 저장소는 *RockSiN: Robust Cross-domain Siamese Network for Periocular Biometrics*의 공식 구현입니다.

---

## 📘 개요

RockSiN은 **안경 착용 여부**로 인해 발생하는 domain shift 문제를 해결하기 위해 설계된  
차세대 periocular biometrics 모델이다.

기존 방법은 안경으로 생성되는 반사(glare), 프레임 가림, 렌즈 왜곡 등을  
단순히 *노이즈처럼 제거하려고* 하기 때문에 cross-domain 상황에서 성능이 급격히 떨어진다.

RockSiN은 이러한 한계를 해결하기 위해 다음 두 가지 특징을 **명시적으로 분리하여 학습**한다.

- **Siamese Branch** → 개인 고유의 생체적 특징(physiological features)  
- **Diff Branch** → 안경 착용 변화로 생기는 domain discrepancy  
- **Fusion Layer** → 두 정보를 결합하여 강인한 매칭 점수 산출

이를 통해 현실적인 환경(HMD, VR/AR, glasses/no-glasses)에서도 안정적인 인식 성능을 보인다.

---

## ✨ 핵심 기여

- Siamese + Diff Branch를 결합한 **듀얼 브랜치 구조** 제안  
- 안경 착용으로 인한 도메인 차이를 **노이즈가 아닌 정보로 활용**  
- AffectiVR 기준 **98.36% AUC, 6.84% EER** 달성  
- Grad-CAM, t-SNE, Linear Probe 실험으로  
  각 branch가 명확히 다른 정보를 학습함을 **해석 가능하게 증명**  
- 실시간 HMD/VR 인증에서도 활용 가능한 경량·고성능 구조

---

## 📊 성능 요약 (AffectiVR)

| Backbone     | Siamese         | Diff             | **RockSiN (Ours)** |
|--------------|------------------|------------------|---------------------|
| MobileNetV3  | 92.60 / 14.39    | 95.97 / 11.84    | **98.36 / 6.84**    |
| ResNet18     | 92.12 / 15.44    | 95.15 / 12.45    | **97.63 / 8.60**    |
| SHViT-s2     | 88.78 / 19.17    | 93.22 / 15.12    | **94.47 / 13.35**   |

값은 AUC(%) / EER(%) 기준.

---

## 🧠 아키텍처 요약

RockSiN 구조는 크게 세 부분으로 구성된다.

1. **Siamese Branch**
   - 두 입력 이미지로부터 생체 특징을 추출  
   - 입력 간 차이의 절댓값(|f1 - f2|)을 특징으로 사용  

2. **Diff Branch**
   - 두 영상의 pixel-wise difference(x1 - x2)를 입력으로 사용  
   - 안경 착용 여부에 따른 domain discrepancy 학습  

3. **Fusion Layer**
   - 두 branch의 embedding을 concat해 최종 similarity 산출  



