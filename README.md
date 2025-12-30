🖥️ Smart PC Builder: AI 기반 PC 견적 및 공간 시뮬레이션
"복잡한 PC 견적, AI가 가격 흐름을 예측하고 공간을 시각화합니다." > 사용자의 예산과 책상 환경에 딱 맞는 최적의 컴퓨터 조립을 돕는 웹 서비스입니다.

📅 프로젝트 소개 (Project Overview)
PC 부품 시장은 가격 변동성이 크고, 부품 간 호환성(규격, 병목 현상) 확인이 어렵습니다. 이 프로젝트는 머신러닝/딥러닝 기술을 활용하여 단순 최저가 비교를 넘어 **'언제 사는 것이 가장 이득인지(Timing)'**와 **'내 책상에 맞는지(Space)'**를 분석해줍니다.

🎯 핵심 목표

AI 가격 지능: 5년치 시세 데이터를 학습해 미래 가격을 예측하고 구매 적기를 추천합니다. 


가성비 분석: 성능 데이터(Benchmark)와 실시간 가격을 결합하여 최적의 부품을 선별합니다. 


공간 시뮬레이션 (Planned): 웹캠을 통한 손 크기 측정 및 데스크 셋업 시각화를 제공합니다. 


🛠️ 기술 스택 (Tech Stack)
AI & Data Analysis (Current Focus)
Language: Python 3.9

Deep Learning: TensorFlow, Keras (LSTM Model)

Machine Learning: Scikit-learn (Preprocessing, Evaluation)


Data Processing: Pandas, NumPy, Regular Expression 

Visualization: Matplotlib, Seaborn, Streamlit (Prototyping)

Web Service (Planned Architecture)

Frontend: React, Three.js (3D Rendering) 


Backend: Java Spring Boot (Main), FastAPI (AI Serving) 


Database: MariaDB 

📊 현재 진행 상황 (Current Progress)
✅ Phase 1: 시계열 기반 부품 가격 예측 시스템 (완료)

[1주차 목표 달성] 주요 PC 부품(VGA, CPU, RAM)의 과거 시세 데이터를 크롤링 및 전처리하여, LSTM 기반의 가격 예측 모델을 구축했습니다. 

1. 데이터 파이프라인 구축

수집: 다나와 등 주요 마켓의 일별 최저가 데이터 수집 (CSV). 

전처리 (Preprocessing):

정규표현식(Regex)을 활용한 모델명/날짜/가격 정보 정밀 추출. 

이상치(Outlier) 제거 및 결측치 보간.

MinMaxScaler를 활용한 데이터 정규화 (0~1 Scaling).

노이즈 제거를 위한 3일 이동평균(Moving Average) 적용.

2. AI 모델링 (LSTM)

Model Selection: 단순 선형 회귀(Linear Regression) 대신, 시계열 데이터의 **비선형적 패턴(순서와 흐름)**을 학습하기 위해 LSTM(Long Short-Term Memory) 딥러닝 모델 채택. 

Architecture:

Input: 30일치 시퀀스 데이터 (Window Size = 30)

Hidden: LSTM(128) → Dropout(0.2) → LSTM(64)

Output: 익일 예측 가격 (Dense)

Performance:

R2 Score(결정계수): 0.93+ 달성 (RTX 4060 기준)

Huber Loss 함수를 사용하여 가격 폭등(Outlier)에 강건한 모델 구현.

3. 서빙 및 시각화
학습된 모델(.h5)과 스케일러(.pkl)를 분리 저장하여 경량화된 추론(Inference) 환경 구축.

Streamlit/Flask를 활용하여 특정 모델(예: RTX 4060) 선택 시 90일 후 가격 추세 그래프 시각화.

🚀 트러블 슈팅 (Troubleshooting)
1. 메모리 부족 문제 (OOM)
문제: 16GB RAM 환경에서 전체 데이터셋 로드 및 학습 시 Memory Out of Bound 발생.

해결:

환경 분리: 모델 학습(Training)은 고사양 워크스테이션(32GB RAM)에서 수행하고, 추론(Inference) 서버는 학습된 모델 파일만 로드하도록 구조 분리.

배치 처리: train_all.py 스크립트를 통해 모델별로 순차 학습 후 메모리를 해제하는 파이프라인 구축.

2. 다양한 제조사 데이터의 파편화
문제: 동일 칩셋(예: RTX 4060)이라도 제조사(ASUS, MSI 등)별 데이터가 적어 개별 학습 시 과적합 발생.

해결: 칩셋 단위로 데이터를 통합하여 전체적인 가격 흐름(Trend)을 학습시키고, 사용자에게는 '칩셋 평균 추세'를 제공하는 방식으로 정확도 확보.

🔮 향후 로드맵 (Roadmap)
[x] Week 1: PC 부품 시계열 데이터 전처리 및 LSTM 가격 예측 모델 구현.

[ ] Week 2: 웹캠 기반 손 크기 측정(MediaPipe) 및 부품 규격 매칭. 

[ ] Week 3: RAG 기반 하드웨어 상담 챗봇(LLM) 구축. 

[ ] Final: React + Spring Boot 연동 및 Three.js 비주얼라이저 통합.
