import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib
import os
import re
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. 페이지 구성
st.set_page_config(page_title="VGA 지능형 분석 시스템", layout="wide")
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# st.cache_resource : 한번만 실행되며 결과값(모델)을 메모리에 저장
@st.cache_resource

# 파일이 실제로 존재하면 모델과 스케일러를 로드, 없으면 반환 x
def load_essentials():
    if os.path.exists("vga_model.h5") and os.path.exists("vga_scaler.pkl"):
        return load_model("vga_model.h5"), joblib.load("vga_scaler.pkl")
    return None, None

# VGA 선택 창 제작을 위한 샘플링
# 경로가 존재하지 않을 경우 빈 리스트를 반환
def get_model_list(path):
    if not os.path.exists(path): return [], []
    
    # 폴더 내의 모든 .csv 파일 목록을 가져와 정렬
    files = sorted([f for f in os.listdir(path) if f.endswith('.csv')])
    try:
        # 가장 최근 파일을 샘플로 읽어서 제품명 목록 제작(files[-1] : 최신, flies[0] : 가장 오래된것)
        # utf-8로 인코딩, 실패하면 cp949(한글 윈도우 기본)으로 읽기
        try: df_sample = pd.read_csv(os.path.join(path, files[-1]), encoding='utf-8')
        except: df_sample = pd.read_csv(os.path.join(path, files[-1]), encoding='cp949')
        
        # 핵심 로직(정규표현식)
        def cleaner(name):
            # RTX or RX 뒤 숫자 4개, 그리고 뒤에 붙는 Ti/Super 등을 조회
            match = re.search(r'(RTX|RX)\s?\d{4}\s?(Ti|SUPER|XT|XTX|GRE)?', name, re.I)
            return match.group(0).strip() if match else None
        
        # 'Name' 컬럼에 함수 적용 -> 중복 제거 -> 리스트로 변환 -> 정렬
        return sorted(df_sample['Name'].apply(cleaner).dropna().unique().tolist()), files
    except: return [], files

# 데이터 경로
VGA_PATH = r'D:\minwoo\project\Virtual-Build-PC\last_data\VGA_Total'

# pkl(머신러닝), d5(딥러닝) 모델 로드
model_ai, scaler_ai = load_essentials()

# VGA_PATH 내의 목록에 있는 csv, 모델 가져오기
model_list, all_files = get_model_list(VGA_PATH)

# 사이드 바 선택 상자(기본으로 RTX 4060 선택)
selected_model = st.sidebar.selectbox("🎮 그래픽카드 선택", model_list, index=model_list.index("RTX 4060") if "RTX 4060" in model_list else 0)

@st.cache_data  # 핵심 데이터 캐싱
def load_data(target):
    all_data = []
    for f in all_files:     # 모든 CSV 파일을 순회
        df_tmp = None       # df_tmp를 초기화 시켜 데이터 중복을 방지 
        
        # UTF-8로 읽기 실패할 경우 cp494로 다시 읽기
        for enc in ['utf-8', 'cp949']:
            try: df_tmp = pd.read_csv(os.path.join(VGA_PATH, f), encoding=enc); break
            except: continue
            
        # 데이터 가공 로직(vga.csv 파일 읽기 성공 한 경우 실행)
        if df_tmp is not None:
            
            # df_tmp의 'name' 컬럼에 대해 target이 포함된 행만 찾기(이름이 없을 경우 제외, 대소문자 구분 x)
            rows = df_tmp[df_tmp['Name'].str.contains(target, na=False, case=False)]
            # XXXX-XX-XXXX 형식(날짜)인 데이터만 수집
            date_cols = [c for c in df_tmp.columns if re.match(r'\d{4}-\d{2}-\d{2}', c)] # re.정규표현식
            
            # 날짜 별로 반복
            for col in date_cols:
                # .astype(str): 문자열 -> 콤마 제거 -> 정규표현식으로 숫자만 뽑기 -> 추출된 숫자 문자열을 숫자로만 처리
                p = pd.to_numeric(rows[col].astype(str).str.replace(',', '').str.extract('(\d+)')[0], errors='coerce')
                # 가격오류 방지(10000원 이상 제품만 수집)
                valid = p[p > 10000]
                # 날짜 뒤 시간(2024-01-01 14:00)의 경우 공백 삭제 이후 날짜만 수집 
                # 해당 날짜에 수집된 가격들의 평균 계산 및 리스트에 {날짜, 평균가격} 형태로 저장
                if not valid.empty: all_data.append({'Date': col.split(' ')[0], 'Price': valid.mean()})
    # 데이터 유무 확인
    if not all_data: return None
    
    # 리스트 형태의 데이터를 표(DataFrame)로 변환 후 날짜로 묶어 가격의 평균만 남기고 그룹으로 다시 묶기
    df = pd.DataFrame(all_data).groupby('Date')['Price'].mean().reset_index()
    # 글자 형태를 날짜(시간)으로 분석 후 오름차순(과거 -> 현재)정렬
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date_dt')    
    # EDA 분석용(연도, 월, 요일 각각 추출)
    df['Year'], df['Month'], df['DayOfWeek'] = df['Date_dt'].dt.year, df['Date_dt'].dt.month, df['Date_dt'].dt.dayofweek
    # 오늘 날짜 - 시작일을 계산해 시간 간격 계산 후 숫자로만 뽑아내기(정수)
    df['DaysFromStart'] = (df['Date_dt'] - df['Date_dt'].min()).dt.days
    # 노이즈 제거
    df['Price_Raw'] = df['Price']   # 원래 가격 데이터 백업
    df['Price_Smooth'] = df['Price'].rolling(window=3, min_periods=1).mean() #3일 이동 평균계산 
    # 최종 정리된 표(DataFrame) 반환
    return df

# 유저가 선택한 모델 데이터 로드
df_final = load_data(selected_model)

# 유저가 선택한 모델의 데이터가 존재할 경우 
if df_final is not None:
    st.title(f"📊 {selected_model} 탐색적 데이터 분석(EDA) & 예측 Overview")

    # --- 섹션 1: AI 모델 성능 지표 (R2, MAE, MSE, RMSE) ---
    st.header("1. 모델 성능 및 정확도 지표")
    # 30일 동안의 가격 흐름 참고
    SEQ_LENGTH = 30
    # 모든 가격을 0 ~ 1 사이의 소수로 변경
    scaled_data = scaler_ai.transform(df_final[['Price_Smooth']])
    # 데이터가 30개 이상일 경우 만 실행
    if len(scaled_data) > SEQ_LENGTH:
        # Sliding Window 
        # scaled_data[[0 : 30], [1 : 31], [2 : 32]...] 형태의 문제 생성
        X_val = np.array([scaled_data[i:i+SEQ_LENGTH] for i in range(len(scaled_data)-SEQ_LENGTH)])
        # X_val 문제들을 AI에게 제공 및 정답 y_pred(0 ~ 1범위) 추출 / verbose: 결과 출력 과정 표시
        y_pred = scaler_ai.inverse_transform(model_ai.predict(X_val, verbose=0))
        # 1 ~ 30일차의 가격을 제외한 31일차부터의 AI 예측 가격을 모으기
        y_actual = df_final['Price_Smooth'].values[SEQ_LENGTH:]
        
        # 웹화면을 가로로 4등분
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("결정계수 (R²)", f"{r2_score(y_actual, y_pred):.4f}")
        m2.metric("평균 절대 오차 (MAE)", f"{mean_absolute_error(y_actual, y_pred):,.0f}원")
        m3.metric("평균 제곱 오차 (MSE)", f"{mean_squared_error(y_actual, y_pred):,.0f}")
        m4.metric("평균 제곱근 오차 (RMSE)", f"{np.sqrt(mean_squared_error(y_actual, y_pred)):,.0f}원")

    # --- 섹션 2: 주요 변수별 분포 (EDA Overview) ---
    st.markdown("---")
    st.header("2. 주요 변수별 데이터 분포 (EDA)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💰 가격 분포 히스토그램")
        fig1, ax1 = plt.subplots()
        sns.histplot(df_final['Price_Raw'], kde=True, color='skyblue', ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader("📅 요일별 데이터 빈도")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df_final, x='DayOfWeek', palette='viridis', ax=ax2)
        st.pyplot(fig2)
    with col3:
        st.subheader("📅 월별 데이터 빈도")
        fig3, ax3 = plt.subplots()
        sns.countplot(data=df_final, x='Month', palette='magma', ax=ax3)
        st.pyplot(fig3)

    col4, col5 = st.columns(2)
    with col4:
        st.subheader("🔗 변수 간 상관관계 (5x5)")
        fig4, ax4 = plt.subplots()
        sns.heatmap(df_final[['Price','Year','Month','DayOfWeek','DaysFromStart']].corr(), annot=True, cmap='coolwarm', ax=ax4)
        st.pyplot(fig4)
    with col5:
        st.subheader("📦 시세 이상치 분석 (Boxplot)")
        fig5, ax5 = plt.subplots()
        sns.boxplot(x=df_final['Price_Raw'], color='salmon', ax=ax5)
        st.pyplot(fig5)

    # --- 섹션 3: 시세 추이 및 미래 예측 ---
    st.markdown("---")
    st.header("3. 시세 추이 및 미래 90일 예측")
    tab_past, tab_future = st.tabs(["과거 학습 데이터 비교", "미래 시세 예측"])
    
    with tab_past:
        fig_v, ax_v = plt.subplots(figsize=(12, 5))
        ax_v.plot(df_final['Date_dt'].values[SEQ_LENGTH:], y_actual, label='실제값')
        ax_v.plot(df_final['Date_dt'].values[SEQ_LENGTH:], y_pred, label='AI 예측', linestyle='--')
        ax_v.legend(); st.pyplot(fig_v)
        
    with tab_future:
        last_seq = scaled_data[-SEQ_LENGTH:]
        future_preds = []
        for _ in range(90):
            nv = model_ai.predict(last_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
            future_preds.append(nv[0]); last_seq = np.append(last_seq[1:], nv, axis=0)
        future_prices = scaler_ai.inverse_transform(future_preds)
        future_dates = [df_final['Date_dt'].max() + timedelta(days=i) for i in range(1, 91)]
        
        fig_f, ax_f = plt.subplots(figsize=(12, 5))
        ax_f.plot(df_final['Date_dt'].iloc[-60:], df_final['Price_Raw'].iloc[-60:], label='Past')
        ax_f.plot(future_dates, future_prices, label='AI Forecast', color='red')
        ax_f.legend(); st.pyplot(fig_f)

else:
    st.error("데이터 로드 실패")