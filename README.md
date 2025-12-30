# 🖥️ Smart PC Builder: AI 기반 PC 견적 및 공간 시뮬레이션

![Project Status](https://img.shields.io/badge/Status-In_Progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **"복잡한 PC 견적, AI가 가격 흐름을 예측하고 공간을 시각화합니다."**
> 사용자의 예산과 책상 환경에 딱 맞는 최적의 컴퓨터 조립을 돕는 웹 서비스입니다.

<br/>

---

<details>
<summary><b>📚 1. 프로젝트 소개 (Project Overview) - 클릭하여 펼치기</b></summary>
<br/>

### 💡 기획 의도
PC 부품 시장은 가격 변동성이 크고, 부품 간 호환성(규격, 병목 현상) 확인이 어렵습니다. 이 프로젝트는 머신러닝/딥러닝 기술을 활용하여 단순 최저가 비교를 넘어, 사용자가 **'언제 사는 것이 가장 이득인지(Timing)'**와 **'내 책상에 맞는지(Space)'**를 과학적으로 분석해주는 솔루션입니다.

### 🎯 핵심 목표
* **💰 AI 가격 지능:** 5년치 시세 데이터를 학습해 미래 가격을 예측하고 구매 적기를 추천합니다.
* **📊 가성비 분석:** 벤치마크 성능 데이터와 실시간 가격을 결합하여 최적의 부품(Price/Perf)을 선별합니다.
* **📏 공간 시뮬레이션 (Planned):** 웹캠을 통한 손 크기 측정 및 데스크 셋업 시각화를 제공합니다.
* **🤖 하드웨어 챗봇 (Planned):** 초보자도 쉽게 이해할 수 있는 RAG 기반 AI 상담을 제공합니다.

</details>

<br/>

<details>
<summary><b>🛠️ 2. 기술 스택 (Tech Stack) - 클릭하여 펼치기</b></summary>
<br/>

### **AI & Data Analysis (Core)**
| Category | Stack | Description |
| --- | --- | --- |
| **Language** | Python 3.9 | 데이터 분석 및 모델링 메인 언어 |
| **Deep Learning** | TensorFlow, Keras | LSTM 시계열 예측, CNN 객체 인식 |
| **Machine Learning** | Scikit-learn, XGBoost | 데이터 전처리, 회귀 분석, 성능 평가 |
| **Computer Vision** | OpenCV, MediaPipe | (예정) 손 크기 측정, 객체 탐지 |
| **LLM** | LangChain, OpenAI API | (예정) RAG 기반 하드웨어 상담 봇 |

### **Application & Serving**
* **Frontend/Dashboard:** Streamlit (현재 프로토타입), React (추후 고도화 예정)
* **Backend:** Flask/FastAPI (Model Serving)
* **Database:** Pandas(CSV) -> MariaDB (추후 마이그레이션 예정)

</details>

<br/>

<details open>
<summary><b>📅 3. 개발 일정 및 진행 상황 (Schedule & Progress) - 클릭하여 펼치기</b></summary>
<br/>

> **전체 개발 기간:** 2024.12.22 ~ 2025.01.20 (4주 예상)

### ✅ Week 1: 시계열 기반 부품 가격 예측 시스템 (완료)
**기간:** 2024.12.22 ~ 2024.12.30

* **목표:** 주요 부품(VGA, CPU, RAM) 데이터 파이프라인 구축 및 가격 예측 모델 개발.
* **상세 내용:**
    * [x] 다나와/쇼핑몰 시세 데이터 크롤링 및 정규표현식(`Regex`) 전처리.
    * [x] 이상치(Outlier) 제거 및 `MinMaxScaler` 정규화.
    * [x] **LSTM(Long Short-Term Memory)** 기반 가격 예측 모델 학습 (R2 Score 0.93+ 달성).
    * [x] `Streamlit` 웹 대시보드 연동 및 90일 미래 가격 추세 시각화.
    * [x] 모델 경량화(`.h5`, `.pkl` 분리) 및 서빙 최적화.

### 🔜 Week 2: 비전 AI 기반 신체 측정 및 호환성 체크 (진행 예정)
**기간:** 2024.12.31 ~ 2025.01.06

* **목표:** 웹캠을 활용한 사용자 손 크기 측정 및 주변기기 매칭 기능 구현.
* **계획:**
    * [ ] `MediaPipe Hands`를 활용한 손가락 랜드마크(21개 포인트) 추출.
    * [ ] 픽셀-mm 변환 알고리즘(Calibration) 구현.
    * [ ] 마우스/키보드 DB 구축 및 사용자 손 크기 대비 추천 로직(Fit-Check) 개발.
    * [ ] `OpenCV`를 활용한 부품 실물 크기 체감 필터 구현.

### 🔜 Week 3: LLM 기반 하드웨어 상담 챗봇 구축 (진행 예정)
**기간:** 2025.01.07 ~ 2025.01.13

* **목표:** RAG(검색 증강 생성) 기술을 도입하여 전문적인 하드웨어 상담 제공.
* **계획:**
    * [ ] PC 부품 매뉴얼 및 호환성 가이드 데이터 벡터화(Vector DB).
    * [ ] `LangChain` 프레임워크를 활용한 Q&A 봇 구축.
    * [ ] "병목 현상 해결해줘", "이 케이스에 그래픽카드 들어가?" 등 자연어 처리.

### 🔜 Final: 웹 서비스 통합 및 고도화 (진행 예정)
**기간:** 2025.01.14 ~ 2025.01.20

* **목표:** 전체 기능을 하나의 웹 서비스로 통합하고 UI/UX 개선.
* **계획:**
    * [ ] Streamlit 대시보드 UI 개선 및 배포.
    * [ ] Three.js (Optional) 활용 데스크 셋업 3D 뷰어 프로토타이핑.
    * [ ] 최종 포트폴리오 정리 및 시연 영상 제작.

</details>

<br/>

<details>
<summary><b>🚀 4. 트러블 슈팅 (Dev Log) - 클릭하여 펼치기</b></summary>
<br/>

### 1. 메모리 부족 (OOM) 문제 해결
* **이슈:** 16GB RAM 환경에서 전체 부품 데이터를 로드하고 LSTM 학습 시 `Memory Out of Bound` 에러 발생.
* **해결:**
    1.  **환경 이원화:** 학습(Training)은 고사양 워크스테이션(32GB)에서 수행, 추론(Inference)은 학습된 가중치 파일(`.h5`)만 로드하여 저사양 PC에서도 구동 가능하게 변경.
    2.  **배치 학습:** 모든 데이터를 한 번에 올리지 않고, `train_all.py` 스크립트를 통해 모델별로 순차 학습(Iterative Training) 후 메모리를 비우는 방식 적용.

### 2. 데이터 불균형 및 파편화
* **이슈:** 특정 제조사(예: Palit, Zotac)의 데이터가 부족하여 개별 모델 학습 시 과적합(Overfitting) 발생 및 예측 그래프가 끊기는 현상.
* **해결:** 제조사별로 쪼개지 않고, **칩셋(Chipset, 예: RTX 4060) 단위로 데이터를 통합**하여 전체적인 가격 트렌드(Trend)를 학습시킴. 이를 통해 데이터 부족 문제를 해결하고 일반화된 성능 확보.

</details>

<br/>

<details>
<summary><b>💻 5. 실행 방법 (How to Run) - 클릭하여 펼치기</b></summary>
<br/>

```bash
# 1. 저장소 복제
git clone [https://github.com/your-username/smart-pc-builder.git](https://github.com/your-username/smart-pc-builder.git)

# 2. 가상 환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. 필수 라이브러리 설치
pip install -r requirements.txt

# 4. Streamlit 앱 실행
streamlit run app.py
