# 🖥️ Smart PC Builder: AI 기반 PC 견적 및 하드웨어 핏(Fit) 솔루션

![Project Status](https://img.shields.io/badge/Status-In_Progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-0099CC?style=flat-square&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

> **"가격은 AI가 예측하고, 호환성은 비전 AI가 맞춰줍니다."**
> 데이터 사이언스와 딥러닝 비전 기술을 결합하여, 최적의 구매 타이밍과 내 손에 꼭 맞는 장비를 추천하는 차세대 PC 빌딩 플랫폼입니다.

<br/>

---

<details>
<summary><b>📚 1. 프로젝트 소개 (Project Overview) - 클릭하여 펼치기</b></summary>
<br/>

### 💡 기획 의도
PC 조립 시장의 불투명한 가격 변동과 온라인 구매 시 주변기기(마우스 등)의 그립감을 알 수 없는 문제를 해결하기 위해 기획되었습니다. 시계열 예측 AI로 '구매 적기'를 알려주고, 비전 AI로 사용자의 신체를 측정해 '실패 없는 구매'를 돕습니다.

### 🎯 핵심 기능
1.  **💰 AI 가격 지능 (Time-Series):** 5년치 부품 시세 데이터를 LSTM으로 학습해 미래 가격 흐름을 예측.
2.  **🖐️ AI 핸드 핏 (Vision DL):** 웹캠에 손을 비추면 손 크기(길이/너비)를 정밀 측정하고, 내 손에 딱 맞는 인생 마우스를 추천.
3.  **🤖 하드웨어 봇 (LLM):** "배그 풀옵션 견적 짜줘" 같은 자연어 질문에 답하는 RAG 기반 챗봇.

</details>

<br/>

<details>
<summary><b>🛠️ 2. 기술 스택 (Tech Stack) - 클릭하여 펼치기</b></summary>
<br/>

### **AI & Data Analysis (Core)**
| Category | Stack | Description |
| --- | --- | --- |
| **Language** | Python 3.9 | 메인 개발 언어 |
| **Price Prediction** | TensorFlow, LSTM | 시계열 부품 가격 예측 모델링 |
| **Computer Vision** | **MediaPipe, OpenCV** | 손가락 랜드마크 추출 및 길이 측정 |
| **Data Processing** | Pandas, NumPy | 대용량 하드웨어 데이터 전처리 |

### **Application & Serving**
* **Frontend/App:** Streamlit (프로토타입), React (예정)
* **Serving:** Flask / FastAPI (AI 모델 API 서버)
* **Database:** Pandas(File System) -> MariaDB (마이그레이션 예정)

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
    * [x] 모델 경량화 및 `models` 폴더 구조화 완료.

### 🔜 Week 2: 비전 AI 기반 '내 손에 딱 맞는 마우스' 추천 (진행 예정)
**기간:** 2024.12.31 ~ 2025.01.06

* **목표:** 웹캠을 활용한 사용자 손 크기 측정(Hand Measurement) 및 게이밍 기어 매칭 시스템 구현.
* **핵심 기술:** `MediaPipe Hands` (Google), `OpenCV`, `Streamlit-Webrtc`
* **상세 계획:**
    * [ ] **데이터셋 구축:** EloShapes, Rtings 등에서 마우스 실측 사이즈(L/W/H) 및 쉐입 이미지 크롤링 (DB화).
    * [ ] **핸드 트래킹 구현:** MediaPipe를 활용해 손바닥/손가락 21개 랜드마크 실시간 추출.
    * [ ] **캘리브레이션(Calibration):** 화면상 픽셀(px) 단위를 실제 mm 단위로 변환하는 알고리즘 개발 (동전/카드 등 기준 물체 활용 고려).
    * [ ] **Fit-Check 알고리즘:** 사용자 손 크기 대비 최적의 마우스(팜/클로/핑거팁 그립) 추천 로직 구현.
    * [ ] **AR 시각화:** 화면 속 내 손 위에 마우스 쉐입(Outline)을 1:1 비율로 오버레이하여 크기 체감 기능 제공.

### 🔜 Week 3: LLM 기반 하드웨어 상담 챗봇 구축 (진행 예정)
**기간:** 2025.01.07 ~ 2025.01.13

* **목표:** RAG(검색 증강 생성) 기술을 도입하여 전문적인 하드웨어 상담 제공.
* **계획:**
    * [ ] PC 부품 매뉴얼 및 호환성 가이드 데이터 벡터화(Vector DB).
    * [ ] `LangChain` 프레임워크를 활용한 Q&A 봇 구축.
    * [ ] 자연어 처리: "병목 현상 없는 조합 추천해줘", "이 케이스에 4090 들어가?" 등 질문 대응.

### 🔜 Final: 웹 서비스 통합 및 고도화
**기간:** 2025.01.14 ~ 2025.01.20

* **목표:** 전체 기능(가격예측 + 핸드핏 + 챗봇)을 하나의 웹 서비스로 통합.
* **계획:**
    * [ ] Streamlit 멀티페이지 구성 및 UI/UX 개선.
    * [ ] 통합 테스트 및 버그 수정.
    * [ ] 최종 포트폴리오 정리 및 시연 영상(Demo) 제작.

</details>

<br/>

<details>
<summary><b>🚀 4. 트러블 슈팅 (Dev Log) - 클릭하여 펼치기</b></summary>
<br/>

### 1. 메모리 부족 (OOM) 문제 해결
* **이슈:** 16GB RAM 환경에서 전체 부품 데이터를 한 번에 로드 시 `Memory Out of Bound` 에러 발생.
* **해결:**
    1.  **모듈화:** 학습 스크립트를 부품별(`vga.ipynb`, `cpu.ipynb`...)로 분리하여 프로세스 종료 시 메모리가 반환되도록 구조 변경.
    2.  **경량화:** 추론(Inference) 단계에서는 무거운 학습 데이터 없이 `.h5`(모델)와 `.pkl`(스케일러) 파일만 로드하도록 `app.py` 최적화.

### 2. 경로 인식 오류 (FileNotFound)
* **이슈:** `app.py` 실행 시 모델 파일과 데이터셋 경로를 찾지 못하는 문제 발생. 윈도우/리눅스 간 경로 구분자 차이 및 상대 경로 문제.
* **해결:** `os.path.join` 및 절대 경로(`PROJECT_ROOT`) 설정을 도입하여 어떤 환경에서도 안정적으로 파일을 로드하도록 수정. 또한, 모델 파일명이 불일치할 경우를 대비해 '범용 파일명'을 2순위로 찾는 Fallback 로직 구현.

</details>

<br/>

<details>
<summary><b>💻 5. 실행 방법 (How to Run) - 클릭하여 펼치기</b></summary>
<br/>

```bash
# 1. 저장소 복제
git clone [https://github.com/your-username/smart-pc-builder.git](https://github.com/your-username/smart-pc-builder.git)

# 2. 필수 라이브러리 설치
pip install -r requirements.txt
# (MediaPipe, TensorFlow, Streamlit 등이 포함되어 있어야 함)

# 3. Streamlit 앱 실행
streamlit run src/app.py
