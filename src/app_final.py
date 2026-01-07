import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import joblib
import cv2
import math
import time
import requests
import mediapipe as mp
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. 설정 및 라이브러리 로드
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Smart PC Builder", layout="wide")
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# [스타일] 폰트: 얇고 현대적인 SIMPLEX (유지)
MY_FONT = cv2.FONT_HERSHEY_SIMPLEX 

# [사용자 경로 고정]
PROJECT_ROOT = r'D:\minwoo\project\Virtual-Build-PC'
BASE_DATA_DIR = os.path.join(PROJECT_ROOT, 'Dataset')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
MOUSE_DATA_PATH = os.path.join(BASE_DATA_DIR, 'mouse_specs.csv')

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️ TensorFlow 모듈을 찾을 수 없습니다.")

# -----------------------------------------------------------------------------
# 2. 공통 함수 (로직 100% 유지)
# -----------------------------------------------------------------------------
@st.cache_data
def get_model_list(category):
    folder_map = {"VGA": "VGA_Total", "CPU": "CPU_Total", "RAM": "RAM_Total"}
    target_folder = os.path.join(BASE_DATA_DIR, folder_map.get(category, ""))
    if not os.path.exists(target_folder): return [], target_folder
    files = sorted([f for f in os.listdir(target_folder) if f.endswith('.csv')])
    if not files: return [], target_folder
    try:
        latest = files[-1]
        path = os.path.join(target_folder, latest)
        try: df = pd.read_csv(path, encoding='utf-8')
        except: df = pd.read_csv(path, encoding='cp949')
        def cleaner(name):
            if not isinstance(name, str): return None
            if category == "VGA": match = re.search(r'(RTX|RX|GTX)\s?\d{3,4}\s?(Ti|SUPER|XT|XTX|GRE)?', name, re.I)
            elif category == "CPU": match = re.search(r'(i\d-\d{4,5}[KF]*|Ryzen\s?\d\s?\d{4}[GX]?)', name, re.I)
            elif category == "RAM": match = re.search(r'(DDR\d-\d{4})', name, re.I)
            else: return None
            return match.group(0).strip() if match else None
        if 'Name' in df.columns: return sorted(df['Name'].apply(cleaner).dropna().unique().tolist()), target_folder
        return [], target_folder
    except: return [], target_folder

@st.cache_data
def load_data(folder_path, target_model, category):
    all_data = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for f in files:
        path = os.path.join(folder_path, f)
        df_tmp = None
        for enc in ['utf-8', 'cp949']:
            try: df_tmp = pd.read_csv(path, encoding=enc); break
            except: continue
        if df_tmp is None or 'Name' not in df_tmp.columns: continue
        rows = df_tmp[df_tmp['Name'].str.contains(target_model, na=False, case=False)]
        cols = [c for c in df_tmp.columns if re.match(r'\d{4}-\d{2}-\d{2}', c)]
        for col in cols:
            p = pd.to_numeric(rows[col].astype(str).str.replace(',', '').str.extract('(\d+)')[0], errors='coerce')
            limit = 3000 if category == "RAM" else 10000
            valid = p[p > limit]
            if not valid.empty: all_data.append({'Date': col.split(' ')[0], 'Price': valid.mean()})
    if not all_data: return None
    df = pd.DataFrame(all_data).groupby('Date')['Price'].mean().reset_index()
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date_dt')
    df['Year'] = df['Date_dt'].dt.year
    df['Month'] = df['Date_dt'].dt.month
    df['DayOfWeek'] = df['Date_dt'].dt.dayofweek
    df['Price_Raw'] = df['Price']
    df['Price_Smooth'] = df['Price'].rolling(window=3, min_periods=1).mean()
    return df

@st.cache_data
def load_mouse_data():
    if not os.path.exists(MOUSE_DATA_PATH): return pd.DataFrame()
    try: df = pd.read_csv(MOUSE_DATA_PATH, encoding='utf-8-sig')
    except:
        try: df = pd.read_csv(MOUSE_DATA_PATH, encoding='cp949')
        except: return pd.DataFrame()
    rename_map = {'Manufacturer': 'Brand', 'Grip_Type': 'Grip', 'Image_URL': 'image_url'}
    df.rename(columns=rename_map, inplace=True)
    df.columns = df.columns.str.strip() 
    if 'Brand' in df.columns: df['Brand'] = df['Brand'].astype(str).str.title() 
    if 'Length' in df.columns:
        if 'Price' in df.columns:
            df['Price'] = pd.to_numeric(df['Price'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
        for col in ['Length', 'Width', 'Height']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df['Length'] > 0].copy()
    return df

@st.cache_data(show_spinner=False)
def get_mouse_image_from_url(url):
    """[유지] 흰색 마우스 깨짐 방지 (Contour 방식)"""
    try:
        if not str(url).startswith('http'): return None
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            image_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            if img is None: return None
            if img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                max_cnt = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [max_cnt], -1, 255, thickness=cv2.FILLED)
                img[:, :, 3] = mask
                x, y, w, h = cv2.boundingRect(max_cnt)
                pad = 10
                img = img[max(0, y-pad):min(img.shape[0], y+h+pad), max(0, x-pad):min(img.shape[1], x+w+pad)]
                return img
    except: return None
    return None

def rotate_image_with_matrix(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w, new_h = int((h * sin) + (w * cos)), int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]; M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)), M

def rotate_image(image, angle):
    img, _ = rotate_image_with_matrix(image, angle)
    return img

def overlay_transparent(background, overlay, x, y, overlay_size=None, global_alpha=1.0):
    bg_h, bg_w, _ = background.shape
    if overlay_size is not None: overlay = cv2.resize(overlay, overlay_size)
    h, w, _ = overlay.shape
    if x >= bg_w or y >= bg_h or x + w < 0 or y + h < 0: return background
    bg_x, bg_y = max(x, 0), max(y, 0)
    ol_x, ol_y = max(0, -x), max(0, -y)
    w, h = min(w - ol_x, bg_w - bg_x), min(h - ol_y, bg_h - bg_y)
    if w <= 0 or h <= 0: return background
    overlay_crop = overlay[ol_y:ol_y+h, ol_x:ol_x+w]
    bg_crop = background[bg_y:bg_y+h, bg_x:bg_x+w]
    alpha = (overlay_crop[:, :, 3] / 255.0) * global_alpha
    inv_alpha = 1.0 - alpha
    for i in range(3):
        bg_crop[:, :, i] = (alpha * overlay_crop[:, :, i] + inv_alpha * bg_crop[:, :, i])
    background[bg_y:bg_y+h, bg_x:bg_x+w] = bg_crop
    return background

def nothing(x): pass

# -----------------------------------------------------------------------------
# 3. 메인 로직
# -----------------------------------------------------------------------------
st.sidebar.title("💎 Smart PC Builder")
page_mode = st.sidebar.selectbox("기능을 선택하세요", ["🖥️ 부품 시세 분석", "🖱️ 마우스 가상 피팅룸"])

# =============================================================================
# [MODE 1] 부품 시세 분석 (로직 100% 유지)
# =============================================================================
if page_mode == "🖥️ 부품 시세 분석":
    if not HAS_TF: st.error("❌ TensorFlow 미설치")
    else:
        st.sidebar.header("🛠️ 부품 설정")
        category = st.sidebar.radio("부품 종류", ["VGA", "CPU", "RAM"])
        model_list, folder_path = get_model_list(category)

        if model_list:
            idx = 0
            defaults = {"VGA": "RTX 4060", "CPU": "i5-13400", "RAM": "DDR5-5600"}
            for i, name in enumerate(model_list):
                if defaults.get(category, "") in name: idx = i; break
            selected_model = st.sidebar.selectbox("모델명 선택", model_list, index=idx)
        else: st.error(f"❌ '{category}' 데이터 없음"); st.stop()

        st.title(f"📊 {selected_model} ({category}) 분석")
        st.markdown("---")

        df_final = load_data(folder_path, selected_model, category)
        safe_name = selected_model.replace(" ", "_")
        cat_lower = category.lower()
        path_specific = os.path.join(MODEL_DIR, f"{cat_lower}_{safe_name}_model.h5")
        path_generic = os.path.join(MODEL_DIR, f"{cat_lower}_model.h5")
        final_model_path = path_specific if os.path.exists(path_specific) else (path_generic if os.path.exists(path_generic) else None)
        scaler_candidates = [
            os.path.join(MODEL_DIR, f"{cat_lower}_{safe_name}_scaler.pkl"),
            os.path.join(MODEL_DIR, f"{cat_lower}_scaler.pkl"),
            os.path.join(MODEL_DIR, f"{cat_lower}_model.pkl")
        ]
        final_scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
        has_model = (final_model_path is not None) and (final_scaler_path is not None)

        if df_final is not None:
            st.header("1. 모델 성능 및 정확도")
            if has_model:
                try:
                    model_ai = load_model(final_model_path)
                    scaler_ai = joblib.load(final_scaler_path)
                    SEQ_LENGTH = 30
                    scaled_data = scaler_ai.transform(df_final[['Price_Smooth']])
                    if len(scaled_data) > SEQ_LENGTH:
                        X_val = np.array([scaled_data[i:i+SEQ_LENGTH] for i in range(len(scaled_data)-SEQ_LENGTH)])
                        y_pred = scaler_ai.inverse_transform(model_ai.predict(X_val, verbose=0))
                        y_actual = df_final['Price_Smooth'].values[SEQ_LENGTH:]
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("R² Score", f"{r2_score(y_actual, y_pred):.4f}")
                        m2.metric("MAE", f"{mean_absolute_error(y_actual, y_pred):,.0f}원")
                        m3.metric("MSE", f"{mean_squared_error(y_actual, y_pred):,.0f}")
                        m4.metric("RMSE", f"{np.sqrt(mean_squared_error(y_actual, y_pred)):,.0f}원")
                    else: st.warning("데이터 부족")
                except Exception as e: st.error(f"에러: {e}"); has_model = False
            else: st.info("학습된 모델 없음")

            st.markdown("---"); st.header("2. 주요 변수별 데이터 분포")
            c1, c2, c3 = st.columns(3)
            with c1:
                fig, ax = plt.subplots(); sns.histplot(df_final['Price_Raw'], kde=True, ax=ax, color='skyblue'); st.subheader("💰 가격 분포"); st.pyplot(fig)
            with c2:
                fig, ax = plt.subplots(); sns.countplot(data=df_final, x='DayOfWeek', palette='viridis', ax=ax); st.subheader("📅 요일별 빈도"); st.pyplot(fig)
            with c3:
                fig, ax = plt.subplots(); sns.countplot(data=df_final, x='Month', palette='magma', ax=ax); st.subheader("📅 월별 빈도"); st.pyplot(fig)

            st.markdown("---"); st.header("3. 시세 추이 및 미래 예측")
            tab1, tab2 = st.tabs(["과거 데이터", "미래 예측"])
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df_final['Date_dt'], df_final['Price_Raw'], label='Raw', alpha=0.5)
                ax.plot(df_final['Date_dt'], df_final['Price_Smooth'], label='Trend', color='red')
                ax.legend(); st.pyplot(fig)
            with tab2:
                if has_model:
                    last_seq = scaled_data[-SEQ_LENGTH:]
                    future_preds = []
                    for _ in range(90):
                        nxt = model_ai.predict(last_seq.reshape(1, SEQ_LENGTH, 1), verbose=0)
                        future_preds.append(nxt[0])
                        last_seq = np.append(last_seq[1:], nxt, axis=0)
                    future_prices = scaler_ai.inverse_transform(future_preds)
                    last_date = df_final['Date_dt'].max()
                    future_dates = [last_date + timedelta(days=i) for i in range(1, 91)]
                    fig, ax = plt.subplots(figsize=(12, 5))
                    ax.plot(future_dates, future_prices, color='red', label='Future 90 Days')
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.legend(); st.pyplot(fig)
                    diff = future_prices[-1][0] - future_prices[0][0]
                    if diff < -5000: st.success(f"📉 하락 예상 (-{abs(diff):,.0f}원)")
                    elif diff > 5000: st.warning(f"📈 상승 예상 (+{diff:,.0f}원)")
                    else: st.info("⚖️ 보합세 예상")
                else: st.write("모델 없음")
        else: st.error("데이터 로드 실패")

# =============================================================================
# [MODE 2] 마우스 가상 피팅룸 (파이프라인 원복 + 넓이 오차 추가)
# =============================================================================
elif page_mode == "🖱️ 마우스 가상 피팅룸":
    st.title("🖱️ 마우스 가상 피팅룸 (3-Step Calibration)")

    df_mouse = load_mouse_data()
    if 'camera_on' not in st.session_state: st.session_state['camera_on'] = False
    if 'result_data' not in st.session_state: st.session_state['result_data'] = None

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ 카메라 설정")
    cam_id = st.sidebar.selectbox("📷 카메라 번호", [0, 1, 2, 3], index=0)
    
    if st.session_state['camera_on']:
        st.sidebar.success(f"🟢 [채널 {cam_id}] 연결됨")
    else:
        st.sidebar.info("⚪ 카메라 대기 중")

    st.markdown("### 📝 Step 1: 내 손 실측값 입력")
    st.info("""
    **📏 정확한 측정을 위한 기준 (필독):**
    * **길이(L):** 손목 **관절 중앙(접히는 부분)** ~ **중지 끝**
    * **너비(W):** **검지** 뿌리 관절 ~ **새끼** 뿌리 관절 (**엄지 제외**)
    """)
    c1, c2 = st.columns(2)
    with c1: user_hand_l = st.number_input("📏 실제 손 길이 (mm)", 100, 250, 180)
    with c2: user_hand_w = st.number_input("📏 실제 손 너비 (mm)", 50, 150, 85)

    st.markdown("---")
    st.markdown("### 🖱️ Step 2: 마우스 선택")
    if not df_mouse.empty:
        col_sel1, col_sel2 = st.columns([1, 2])
        with col_sel1:
            brand_list = ["All"] + sorted(df_mouse['Brand'].unique().tolist())
            selected_brand = st.selectbox("제조사 필터", brand_list)
        with col_sel2:
            if selected_brand != "All": filtered_df = df_mouse[df_mouse['Brand'] == selected_brand]
            else: filtered_df = df_mouse
            selected_mouse_name = st.selectbox("모델명 검색", filtered_df['Name'].unique())
        
        selected_mouse_info = df_mouse[df_mouse['Name'] == selected_mouse_name].iloc[0]
        
        col_info, col_img = st.columns([3, 1])
        with col_info:
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("길이", f"{selected_mouse_info['Length']} mm")
            cc2.metric("너비", f"{selected_mouse_info['Width']} mm")
            cc3.metric("가격", f"{int(selected_mouse_info['Price']):,} 원")
        with col_img:
            if pd.notna(selected_mouse_info['image_url']):
                st.image(selected_mouse_info['image_url'], use_column_width=True)
            else: st.text("이미지 없음")

    st.markdown("---")
    st.markdown("### 🎥 Step 3: 가상 피팅 시작")
    st.info("""
    **✨ 사용법:**
    1. **Monitor 10cm:** 컨트롤 창의 슬라이더를 움직여 파란 선을 실제 자의 10cm와 맞추세요.
    2. **Cam Scale:** 청록색 박스에 손을 맞추세요.
    3. **캡처:** 초록색 박스가 뜨면 5초간 유지하세요.
    """)

    btn_text = "🟥 피팅 종료" if st.session_state['camera_on'] else "🟩 가상 피팅 시작 (Start AR)"
    if st.button(btn_text, use_container_width=True):
        st.session_state['camera_on'] = not st.session_state['camera_on']
        if st.session_state['camera_on']: st.session_state['result_data'] = None
        st.rerun()

    if st.session_state['camera_on'] and not df_mouse.empty:
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        time.sleep(0.5) 
        
        if not cap.isOpened():
            st.error(f"🚨 {cam_id}번 카메라 연결 실패.")
            st.sidebar.error("🔴 연결 실패")
            st.session_state['camera_on'] = False
        else:
            window_name = "AR Fitting Mode"
            ctrl_name = "Calibration Panel"
            cv2.namedWindow(ctrl_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(ctrl_name, 400, 500)
            
            cv2.createTrackbar("Monitor 10cm", ctrl_name, 100, 300, nothing)
            cv2.createTrackbar("Cam Scale", ctrl_name, 35, 100, nothing)
            cv2.createTrackbar("Alpha (%)", ctrl_name, 90, 100, nothing)
            cv2.createTrackbar("Angle (+90)", ctrl_name, 90, 180, nothing)
            cv2.createTrackbar("Pos X (+100)", ctrl_name, 100, 200, nothing)
            cv2.createTrackbar("Pos Y (+100)", ctrl_name, 100, 200, nothing)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL); cv2.resizeWindow(window_name, 1280, 720)

            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
            
            fit_start_time = None
            capture_success = False
            final_meas_l = 0; final_meas_w = 0; final_px_per_mm = 0
            img_hand_crop = None; img_mouse_clean = None
            
            hand_angle = 0
            grip_center_rel = (0, 0)

            with mp_hands.Hands(max_num_hands=1, model_complexity=0) as hands:
                current_mouse_img = get_mouse_image_from_url(selected_mouse_info['image_url'])
                ctrl_bg = np.zeros((500, 400, 3), dtype=np.uint8)
                # [스타일] AA 적용 (폰트 SIMPLEX)
                cv2.putText(ctrl_bg, "CONTROLS", (20, 40), MY_FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                while cap.isOpened() and st.session_state['camera_on']:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.resize(frame, (1280, 720)); frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape
                    
                    try:
                        val_monitor = max(0.1, cv2.getTrackbarPos("Monitor 10cm", ctrl_name) / 100.0)
                        val_cam = max(5, cv2.getTrackbarPos("Cam Scale", ctrl_name))
                        val_alpha = cv2.getTrackbarPos("Alpha (%)", ctrl_name)/100.0
                        val_angle = cv2.getTrackbarPos("Angle (+90)", ctrl_name)-90
                        val_x = cv2.getTrackbarPos("Pos X (+100)", ctrl_name)-100
                        val_y = cv2.getTrackbarPos("Pos Y (+100)", ctrl_name)-100
                    except: break

                    # 표준 모니터 해상도 96DPI 기준 10cm 표현할때 필요한 픽셀 수
                    monitor_10cm_px = int(378 * val_monitor)
                    
                    # [스타일] 선 두께 2, AA, Cyan 색상
                    cv2.line(frame, (50, h-50), (50+monitor_10cm_px, h-50), (255, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, "10cm Ruler", (50, h-70), MY_FONT, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
                    
                    t_h, t_w = int((user_hand_l/10)*val_cam*val_monitor), int((user_hand_w/10)*val_cam*val_monitor)
                    cx, cy = w // 2, h // 2
                    tl, br = (cx-t_w//2, cy-t_h//2), (cx+t_w//2, cy+t_h//2)
                    
                    box_color, msg = (200, 200, 200), "Fit Hand Here"
                    
                    # 1. 이미지 색공간 변환 : BRG Image -> RGB(MediaPipe 모델 요구사항)
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 2. 딥러닝 모델 추론
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        for hl in results.multi_hand_landmarks:
                            # 21개 랜드마크의 정규화된 좌표(0.0 ~ 1.0) 픽셀 좌표로 변환
                            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                            
                            # 핵심 랜드마크 추출 (0: 손목, 12: 중지 끝, 5: 검지 뿌리, 17: 새끼 뿌리)
                            p0, p12 = hl.landmark[0], hl.landmark[12]
                            p5, p17 = hl.landmark[5], hl.landmark[17]
                            
                            # 유클리드 거리 공식을 활용한 손의 길이와 너비(pixel 단위) 계산
                            dist_l_px = math.sqrt((p12.x*w-p0.x*w)**2 + (p12.y*h-p0.y*h)**2)
                            dist_w_px = math.sqrt((p17.x*w-p5.x*w)**2 + (p17.y*h-p5.y*h)**2)

                            # Calibration : 픽셀 단위를 실제 물리적 거리로 변환
                            meas_l = (dist_l_px / (val_cam * val_monitor)) * 10
                            meas_w = (dist_w_px / (val_cam * val_monitor)) * 10
                            
                            # 실제 입력한 수치 - AI 측정값 사이 오차 계산
                            diff_l = meas_l - user_hand_l   # 손 길이
                            diff_w = meas_w - user_hand_w   # 손 너비
                            
                            # 오차 범위(5mm)에 따른 텍스트 색상 결정(Pass시 초록)
                            col_l = (0, 255, 0) if abs(diff_l) < 5 else (0, 0, 255)
                            col_w = (0, 255, 0) if abs(diff_w) < 5 else (0, 0, 255)

                            # 실시간 측정 및 오차율 화면 상단에 출력
                            cv2.putText(frame, f"Hand L: {meas_l:.1f}mm (Err: {diff_l:+.1f})", (30, 50), MY_FONT, 0.7, col_l, 1, cv2.LINE_AA)
                            cv2.putText(frame, f"Hand W: {meas_w:.1f}mm (Err: {diff_w:+.1f})", (30, 80), MY_FONT, 0.7, col_w, 1, cv2.LINE_AA)
                            
                            # 존이 중앙 박스에 위치 했는지 확인
                            h_cx, h_cy = (p0.x*w + p12.x*w) // 2, (p0.y*h + p12.y*h) // 2
                            dist_c = math.sqrt((h_cx - cx)**2 + (h_cy - cy)**2)
                            
                            # Width, Length 5mm 이내 and 중앙 정렬 시 5초 카운트 시작
                            if abs(diff_l) < 5 and dist_c < 60: 
                                box_color = (0, 255, 0) # 박스 색상 초록으로 변경
                                if fit_start_time is None: fit_start_time = time.time()
                                elap = time.time() - fit_start_time
                                if 5.0 - elap > 0:
                                    msg = f"Hold: {5.0-elap:.1f}s"  # 남은 시간
                                    bw = int((elap/5.0)*t_w)        # 프로그레스 바 너비 계산
                                    cv2.rectangle(frame, (tl[0], tl[1]-20), (tl[0]+bw, tl[1]-10), (0, 255, 0), -1)
                                else:
                                    # 5초 대기 완료 및 데이터 확정일 경우 캡처 성공
                                    msg, capture_success = "Complete!", True
                                    final_meas_l, final_meas_w = meas_l, meas_w
                                    final_px_per_mm = monitor_10cm_px / 100.0
                                    
                                    # 손목(p0), 중지 너클(P9)사이의 기울기를 구해 마우스 각도 회전
                                    p9 = hl.landmark[9]
                                    dx = p9.x*w - p0.x*w
                                    dy = p9.y*h - p0.y*h
                                    
                                    # atan2를 사용하여 라디안 -> 각도 변환
                                    hand_angle = -90 - math.degrees(math.atan2(dy, dx))
                                    
                                    # 결고 리포트용 손 이미지 자르기
                                    x_list = [lm.x * w for lm in hl.landmark]
                                    y_list = [lm.y * h for lm in hl.landmark]
                                    x_min, x_max = max(0, int(min(x_list))-30), min(w, int(max(x_list))+30)
                                    y_min, y_max = max(0, int(min(y_list))-30), min(h, int(max(y_list))+30)
                                    img_hand_crop = frame[y_min:y_max, x_min:x_max].copy()
                                    img_mouse_clean = current_mouse_img
                                    
                                    # 손가락 뿌리 관절(너클, p5, p9)의 중점을 마우스 합성의 기준점으로 지정
                                    p5_lm = hl.landmark[5]
                                    abs_grip_x = (p5_lm.x*w + p9.x*w) / 2
                                    abs_grip_y = (p5_lm.y*h + p9.y*h) / 2
                                    grip_center_rel = (int(abs_grip_x - x_min), int(abs_grip_y - y_min))
                                    
                            else:
                                # 조건 미달일 경우 카운트다운 초기화 및 안내 매세지 변경
                                fit_start_time = None
                                if abs(diff_l) >= 5: msg = "Size Mismatch"
                                elif dist_c >= 60: msg = "Center Hand"

                            # 실시간 마우스 이미지 합성 : 손 + 마우스 크기 측정 시 
                            if current_mouse_img is not None:
                                # 마우스 제원 길이 기반 픽셀 크기 실시간으로 조정
                                m_h = int(((selected_mouse_info['Length']/10)*val_cam)*val_monitor)
                                r = m_h / current_mouse_img.shape[0]
                                m_w = int(current_mouse_img.shape[1] * r)
                                
                                # 손 각도에 맞춰 마우스 회전
                                rm = rotate_image(cv2.resize(current_mouse_img, (m_w, m_h)), val_angle)
                                # 중지 관절 위치에 마우스 배치 및 오버레이 합성
                                mc = hl.landmark[9]
                                dx, dy = int(mc.x*w - rm.shape[1]//2 + val_x), int(mc.y*h - rm.shape[0]//2 + val_y)
                                
                                # 계산된 좌표 dx, dy에 배경 투명도를 유지하며 합성
                                frame = overlay_transparent(frame, rm, dx, dy, overlay_size=None, global_alpha=val_alpha)
                                display_name = selected_mouse_name if selected_mouse_name.isascii() else "Mouse Model"
                                
                                # [스타일] AA 적용
                                cv2.putText(frame, f"Mouse: {display_name}", (dx, dy-10), MY_FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(frame, f"{selected_mouse_info['Length']}x{selected_mouse_info['Width']}mm", (30, 120), MY_FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

                    # [스타일] 박스 두께 1, AA 적용
                    cv2.rectangle(frame, tl, br, box_color, 1, cv2.LINE_AA)
                    cv2.putText(frame, msg, (cx-100, tl[1]-10), MY_FONT, 0.7, box_color, 1, cv2.LINE_AA)
                    cv2.imshow(window_name, frame); cv2.imshow(ctrl_name, ctrl_bg)
                    
                    # 루프 종료 조건 검사
                    # -q키 입력시 종료
                    # capture_success가 True일 경우(5초 대기 성공 시) 자동 종료 
                    if cv2.waitKey(1) & 0xFF == ord('q') or capture_success: break
                    try:
                        
                        # 예외처리: 사용자가 창을 닫을 경우 예외없이 루프 빠져나가기
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
                        if cv2.getWindowProperty(ctrl_name, cv2.WND_PROP_VISIBLE) < 1: break
                    except: break
            # 자원해제 및 상태 초기화, Streamlit 카메라 세션 off로 변경
            cap.release(); cv2.destroyAllWindows(); st.session_state['camera_on'] = False
            
            # 데이터 후처리 및 최종 저장 
            if capture_success:
                # OpenCV(BGR) 이미지 Streamlit(RGB) 출력용 변환
                if img_hand_crop is not None: img_hand_crop = cv2.cvtColor(img_hand_crop, cv2.COLOR_BGR2RGB)
                if img_mouse_clean is not None: img_mouse_clean = cv2.cvtColor(img_mouse_clean, cv2.COLOR_BGRA2RGBA)

                # 결과 데이터 딕셔너리 형으로 구조화
                # 결과 리포트 데이터 출력 자료형
                st.session_state['result_data'] = {
                    'mouse': selected_mouse_name,   # 마우스 모델명
                    'mouse_len': selected_mouse_info['Length'], # 마우스 실제 길이 스펙
                    'user_l': user_hand_l,  # 사용자 입력 손 길이
                    'user_w': user_hand_w,  # 사용자 입력 손 너비
                    'meas_l': final_meas_l, # AI 측정 손 길이
                    'meas_w': final_meas_w, # AI 측정 손 너비
                    'diff_l': final_meas_l - user_hand_l,   # 길이 측정 오차
                    'diff_w': final_meas_w - user_hand_w,   # 너비 측정 오차
                    'img_hand': img_hand_crop, # 캡처된 실제 손 이미지
                    'img_mouse': img_mouse_clean,   # 가상 피팅에 사용된 마우스 이미지
                    'px_per_mm': final_px_per_mm,   # 1:1 배율 재현을 위한 픽셀 배율
                    'hand_angle': hand_angle,       # 손의 회전 각도 데이터
                    'grip_center': grip_center_rel  # 마우스 합성 기준점 좌표
                }
                st.balloons();  # 축하 애니메이션
                st.rerun()      # 화면 새로고침 후 결과 리포트 레이아웃 출력

    if st.session_state.get('result_data'):
        st.markdown("""
        <style>
        div[data-testid="stImage"] {
            justify-content: center;
            display: flex;
            align-items: center;
            width: 100%;
        }
        div[data-testid="stImage"] > img {
            margin-left: auto;
            margin-right: auto;
            display: block;
        }
        </style>
        """, unsafe_allow_html=True)

        res = st.session_state['result_data']
        st.divider()
        st.success("🎉 측정이 완료되었습니다! (1:1 Real Scale View)")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🖐️ 내 손 측정값", f"{res['meas_l']:.1f} x {res['meas_w']:.1f} mm")
        diff_l, diff_w = res['diff_l'], res['diff_w']
        c2.metric("📏 길이 오차", f"{diff_l:+.1f} mm", delta_color="inverse" if abs(diff_l) > 5 else "normal")
        c3.metric("📏 너비 오차", f"{diff_w:+.1f} mm", delta_color="inverse" if abs(diff_w) > 5 else "normal")

        st.markdown("### 📷 실제 크기 비교 & 가상 그립 (1:1 Scale)")
        st.info("💡 **검증 방법:** 화면 속 파란색 바(Bar)에 실제 자를 대보세요. 바의 길이가 정확히 **5cm**와 일치해야 합니다.")
        
        ratio = res['px_per_mm']
        bar_len_px = int(50 * ratio) 
        bar_img = np.zeros((20, bar_len_px, 3), dtype=np.uint8); bar_img[:] = (0, 0, 255)
        
        # [핵심] 너클 기준 합성 (유지)
        composite_img = res['img_hand'].copy()
        if res['img_mouse'] is not None:
            h_m, w_m = res['img_mouse'].shape[:2]
            target_h = int(res['mouse_len'] * ratio)
            target_w = int(target_h * (w_m / h_m))
            resized_mouse = cv2.resize(res['img_mouse'], (target_w, target_h))
            
            rotated_mouse, M_rot = rotate_image_with_matrix(resized_mouse, res['hand_angle'])
            
            mouse_cx = rotated_mouse.shape[1] // 2
            mouse_cy = rotated_mouse.shape[0] // 2
            dx = res['grip_center'][0] - mouse_cx
            dy = res['grip_center'][1] - mouse_cy

            if rotated_mouse.shape[2] == 4:
                 rotated_mouse = cv2.cvtColor(rotated_mouse, cv2.COLOR_BGRA2RGBA)
            composite_img = overlay_transparent(composite_img, rotated_mouse, int(dx), int(dy), overlay_size=None, global_alpha=0.85)

        st.markdown("---")
        
        c_r1_c1, c_r1_c2 = st.columns(2)
        with c_r1_c1:
            st.markdown("<h5 style='text-align: center;'>1. 내 손 (Captured Hand)</h5>", unsafe_allow_html=True)
            st.image(res['img_hand'], width=res['img_hand'].shape[1])
            st.markdown("<p style='text-align: center;'>▼ 5cm 검증 바</p>", unsafe_allow_html=True)
            st.image(bar_img, width=bar_len_px)
            
        with c_r1_c2:
            st.markdown("<h5 style='text-align: center;'>2. 마우스 실제 크기 (Mouse Size)</h5>", unsafe_allow_html=True)
            if res['img_mouse'] is not None:
                h, w, _ = res['img_mouse'].shape
                target_h_px = int(res['mouse_len'] * ratio)
                target_w_px = int(target_h_px * (w / h))
                if target_w_px > 0:
                    resized_m = cv2.resize(res['img_mouse'], (target_w_px, target_h_px))
                    st.image(resized_m, width=target_w_px)
                st.markdown("<p style='text-align: center;'>▼ 5cm 검증 바</p>", unsafe_allow_html=True)
                st.image(bar_img, width=bar_len_px)
        
        st.markdown("---")

        c_r2_c1, c_r2_c2 = st.columns(2)
        with c_r2_c1:
             st.markdown("<h5 style='text-align: center;'>★ 3. 가상 그립 (Virtual Grip Overlay)</h5>", unsafe_allow_html=True)
             st.image(composite_img, width=composite_img.shape[1])
             st.markdown("<p style='text-align: center;'>▼ 5cm 검증 바</p>", unsafe_allow_html=True)
             st.image(bar_img, width=bar_len_px)
        with c_r2_c2:
             pass

        if st.button("🔄 다시 측정하기", use_container_width=True): st.session_state['result_data'] = None; st.rerun()