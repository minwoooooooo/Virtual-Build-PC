# 🖥️ Smart PC Builder: AI 기반 PC 견적 및 하드웨어 핏(Fit) 솔루션

![Project Status](https://img.shields.io/badge/Status-In_Progress-yellow?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Vision-0099CC?style=flat-square&logo=google&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

> **"가격은 AI가 예측하고, 호환성은 비전 AI가 맞춰줍니다."** > 데이터 사이언스와 딥러닝 비전 기술을 결합하여, 최적의 구매 타이밍과 내 손에 꼭 맞는 장비를 추천하는 차세대 PC 빌딩 플랫폼입니다.

<br/>

---

<details>
<summary><b>📚 1. 프로젝트 소개 (Project Overview)</b></summary>
<br/>

### 💡 기획 의도
PC 조립 시장의 불투명한 가격 변동과 온라인 구매 시 주변기기(마우스 등)의 그립감을 직접 체감할 수 없는 페인 포인트(Pain Point)를 해결하기 위해 기획되었습니다. 시계열 예측 AI로 '구매 적기'를 제안하고, 비전 AI로 사용자의 신체를 실시간 측정해 '실패 없는 하드웨어 구매'를 돕습니다.

### 🎯 핵심 기능
1.  **💰 AI 가격 지능 (Time-Series):** 5년치 부품 시세 데이터를 LSTM으로 학습하여 미래 가격 흐름 예측.
2.  **🖐️ AI 핸드 핏 (Vision AR):** 웹캠을 통해 사용자의 손 크기를 실시간 측정하고, 1:1 실제 크기로 마우스를 가상 피팅.
3.  **🤖 하드웨어 봇 (LLM/RAG):** 맞춤형 하드웨어 호환성 및 견적 상담 (예정).

</details>

<br/>

<details>
<summary><b>🛠️ 2. 기술 스택 (Tech Stack)</b></summary>
<br/>

### **AI & Data Analysis**
| Category | Stack | Description |
| --- | --- | --- |
| **Language** | Python 3.12 | 메인 개발 및 데이터 파이프라인 |
| **Deep Learning** | **TensorFlow, Keras** | 시계열(LSTM) 시세 예측 모델 구축 |
| **Computer Vision** | **MediaPipe, OpenCV** | 21개 랜드마크 추출 및 AR 합성 로직 구현 |
| **Data Processing** | Pandas, NumPy, Scipy | 실측 좌표 기반 기하학 연산 및 데이터 핸들링 |

### **Application & UI**
* **Frontend:** Streamlit (실시간 비전 스트리밍 및 웹 UI)
* **Visualization:** Matplotlib, Seaborn (시세 데이터 시각화)

</details>

<br/>

<details open>
<summary><b>📅 3. 개발 일정 및 진행 상황 (Progress)</b></summary>
<br/>

### ✅ Week 1: 시계열 기반 부품 가격 예측 시스템 (완료)
* **내용:** 과거 시세 데이터를 활용한 LSTM 예측 모델 구축. R2 Score 0.93 달성.

### ✅ Week 2: 비전 AI 기반 '1:1 실측 가상 피팅' (완료)
**핵심 성과:** 웹캠만으로 사용자의 손 크기를 mm 단위로 정밀 측정하고 하드웨어를 가상 매핑함.

* **기술적 상세:**
    * [x] **실시간 핸드 트래킹:** MediaPipe Hands를 사용하여 21개 관절의 3D 랜드마크 실시간 추론.
    * [x] **Geometric Calibration:** 모니터 DPI(96DPI 기준 378px) 보정 계수를 도입하여 픽셀-mm 변환 알고리즘 구현.
    * [x] **강건한 앵커링(Robust Anchoring):** 손가락 끝 대신 변형이 적은 **너클(MCP) 관절 중점**을 기준으로 마우스 합성 중심점 설정.
    * [x] **벡터 기반 회전 매핑:** 손목-너클 벡터의 각도를 `atan2`로 산출하여 마우스가 손의 방향을 실시간 동기화.
    * [x] **1:1 Real-Scale 구현:** DB 내 실제 마우스 제원(Length/Width)을 픽셀 배율에 역산하여 시각적 오차 최소화.

### 🔜 Week 3: LLM 기반 하드웨어 상담 챗봇 구축 (예정)
* **목표:** RAG(검색 증강 생성) 기술을 도입하여 전문적인 하드웨어 호환성 가이드 제공.

</details>

<br/>

<details>
<summary><b>🚀 4. 트러블 슈팅 (Dev Log)</b></summary>
<br/>

### 1. 디바이스 독립적 스케일 보정 (Calibration)
* **이슈:** 사용자마다 모니터 크기와 해상도가 달라 픽셀 밀도가 불일치하는 문제.
* **해결:** 표준 윈도우 96 DPI 수치(378px = 10cm)와 사용자의 수동 보정 계수(`val_monitor`)를 결합한 비례식 알고리즘을 설계하여 어떤 환경에서도 1:1 실측이 가능하게 함.

### 2. 고정밀 이미지 합성 및 배경 제거
* **이슈:** 마우스 이미지의 배경이 흰색일 경우 손가락과 겹칠 때 가독성이 떨어지는 현상.
* **해결:** `cv2.findContours`를 이용해 객체의 정밀 외곽선을 검출하고, Alpha 채널 마스킹을 통해 배경이 완벽히 제거된 AR 오버레이 구현.

### 3. 프로젝트 경로 구조화 문제
* **이슈:** 하위 폴더(`src/`)에서 실행 시 상위의 `Dataset`이나 `models` 폴더를 참조하지 못하는 경로 에러.
* **해결:** `os.path.dirname(os.path.abspath(__file__))`를 활용한 상대 경로 동적 계산 로직을 적용하여 프로젝트 루트를 자동으로 탐색하도록 개선.

### 4. 카메라 원근 왜곡(Perspective Distortion) 분석
* **이슈:** 5cm 검증 바는 정확하나, 렌즈와 피사체 사이의 거리에 의해 마우스가 다소 작아 보이는 원근 현상 발생.
* **분석/대응:** 단일 렌즈의 광각 특성에 의한 현상임을 파악. 향후 MediaPipe의 $z$(Depth) 좌표를 활용한 '거리 기반 가변 스케일 보정 계수'를 도입할 예정.

</details>

<br/>

<details>
<summary><b>💻 5. 실행 방법 (How to Run)</b></summary>
<br/>

```bash
# 1. 저장소 복제
git clone [https://github.com/your-username/smart-pc-builder.git](https://github.com/your-username/smart-pc-builder.git)

# 2. 가상환경 구축 및 라이브러리 설치
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Streamlit 앱 실행
streamlit run src/app.py