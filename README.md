# 🖥️ Virtual Build PC
> 데이터 기반 지능형 하드웨어 빌드 플랫폼

## 📌 Project Overview
단순한 부품 나열식 견적 서비스를 넘어, ML/DL/LLM 기술을 결합하여 성능을 예측하고 사용자의 실제 물리적 환경(책상, 신체)과 부품 규격을 동기화하는 실감형 하드웨어 셋업 솔루션입니다.

---

## 🏗️ 1. 주요 페이지 구성

| 페이지 | 핵심 기능 | 비고 |
| :--- | :--- | :--- |
| [Home] 마켓 보드 | 실시간 하드웨어 & 게임 핫딜 크롤링, 가격 변동 알림 | 데이터 기반 시장 모니터링 |
| [Estimator] AI 견적 | 게임별 옵션 예측(ML), 6개월 가격 추이 그래프, 호환성 체크 | 성능 및 구매 시점 최적화 |
| [Market] 세일 리포트 | 게임 세일 캘린더, 역대 최저가 비교, 가성비(Price/Perf) 분석 | 스마트 소비 가이드 |
| [Visualizer] 현실 대조 | 휴대폰 모델 대비 본체/모니터 크기 애니메이션, 데스크 시뮬레이션 | 2D/3D 실측 비율 배치 |
| [Lab] 신체 피팅 | 웹캠 기반 손 크기 측정(DL), 마우스/기기 가상 그립 피팅 | 개인화된 주변기기 추천 |

---

## 🛠️ 2. 기술 스택 및 역할

### 🧠 AI & Data Science
* ML (Machine Learning): 부품 성능(FPS) 예측, 시계열 데이터 기반 가격 추이 분석, 병목 현상 스코어링.
* DL (Deep Learning): MediaPipe 활용 핸드 랜드마크 추출 및 실측, 객체 인식(Object Detection)을 통한 부품 자동 인식.
* LLM (Large Language Model): RAG(Retrieval-Augmented Generation) 기반 하드웨어 상담 및 오류 진단 가이드.

### 🎨 Graphics & Rendering
* Rendering: WebGL(Three.js) 또는 Canvas API를 활용한 실측 규격 기반의 공간 시뮬레이션 구현.

---

## 🔄 3. 사용자 흐름 (UX Flow)
1. Market: 실시간 게임 세일 및 부품 핫딜 확인.
2. Estimator: 목표 게임 사양에 따른 ML 기반 최적 견적 생성.
3. Analysis: 가격 추이 데이터를 통한 구매 적기 판단.
4. Visualizer: 실제 책상 규격에 부품 배치 및 공간 점유율 시뮬레이션.
5. Lab: 신체 측정 데이터를 활용한 최종 주변기기(마우스/폰) 피팅.

---

## 📊 4. Dataset 계획
* 성능/가격: Danawa(시세), PassMark/3DMark(벤치마크), UserBenchmark.
* 물리 규격: TechPowerUp(VGA), GSMArena(Smartphone), DisplaySpecifications(Monitor).
* 공간 데이터: PC 케이스/키보드 실측 사이즈 제조사 DB 및 사용자 입력 책상 데이터.
* 게임 데이터: Steam Web API, SteamSpy, CheapShark API, SteamDB.

---

## 📅 1주차 진행 상황 (Current Status)
### ✅ 하드웨어 마스터 데이터베이스 구축 및 데이터 엔지니어링
- [ ] Data Scraping: Python(Selenium/BeautifulSoup) 기반 다나와 카테고리별 스펙 크롤러 개발.
- [ ] Data Cleaning: 텍스트 형태의 상세 사양에서 정규표현식을 활용한 실측 mm 데이터 추출.
- [ ] Performance Mapping: 부품명 기준 해외 벤치마크 점수와 국내 시세 데이터 결합.
- [ ] Price Tracking: 시계열 분석을 위한 일일 부품 시세 수집 파이프라인 구축.
