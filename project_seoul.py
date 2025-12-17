import streamlit as st
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from shapely.geometry import Point, shape
import folium
from folium.plugins import HeatMap
import json
from streamlit_folium import st_folium
from scipy.interpolate import make_interp_spline
import google.generativeai as genai
import streamlit.components.v1 as components
import io
import re

# ==============================================================================
# 1. 초기 설정 및 환경 세팅
# ==============================================================================

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
st.set_page_config(page_title="서울 인구 인사이트 대시보드", layout="wide", initial_sidebar_state="expanded")

# Enhanced Tailwind CSS for styling (유지)
components.html("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    body {
        background: linear-gradient(to bottom right, #bfdbfe, #a5b4fc);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #1e293b;
    }
    .stApp { background: transparent; }
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 0.5rem;
        transition: transform 0.3s;
    }
    .card:hover { transform: translateY(-3px); }
    .header { font-size: 2.5rem; color: #312e81; text-align: center; margin: 0.5rem 0; }
    .subheader { font-size: 1.5rem; color: #3730a3; margin: 0.5rem 0; }
    .content { padding: 0.5rem; line-height: 1.5; }
    .map-container { border-radius: 1rem; overflow: hidden; box-shadow: 0 8px 12px rgba(0, 0, 0, 0.1); }
    .section-divider { margin: 0.5rem 0; border-top: 1px solid #a5b4fc; }
    section[data-testid="stSidebar"] { background: linear-gradient(to bottom, #e0f2fe, #bfdbfe); padding: 0.5rem; }
    .stButton>button { margin: 0.2rem 0; }
    .grok-style { color: #4285f4; font-style: italic; }
    .gemini-highlight { background-color: #e0f2fe; padding: 0.5rem; border-left: 4px solid #4285f4; }
    .warning { color: #dc2626; font-weight: bold; }
</style>
""", height=0)

# MongoDB 연결 (Secrets에서 주소 가져오기)
@st.cache_resource
def init_connection():
    # "mongo"와 "host"는 아까 Secrets에 적은 [mongo] host = ... 와 짝꿍입니다.
    return MongoClient(st.secrets["mongo"]["host"])

client = init_connection()

# DB 이름은 원래 쓰시던 거 그대로 유지!
db = client["seoul_population_db"]

# Gemini API 설정
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    GEMINI_CONFIGURED = True
except (KeyError, Exception):
    GEMINI_CONFIGURED = False

# 지역 특성 데이터 분리 (유지보수성 향상)
REGION_TRAITS = {
    "강남구": "서울의 경제 및 문화 중심지, 높은 유동인구와 소비력. IT, 패션, 뷰티 산업이 발달.",
    "종로구": "역사적 유적과 문화재가 풍부한 전통과 현대의 조화. 고령층 비율이 높고 관광객 유입이 많음.",
    "서초구": "고급 주거 및 상업 시설 밀집 지역, 교육열이 높고 법조계 종사자가 많음.",
    "마포구": "젊은층 유입이 활발한 트렌디한 지역, 홍대 문화와 스타트업 생태계가 공존.",
    "성북구": "전통적인 주거 지역으로 조용하고 안정적. 대학가 인접 지역은 젊은층 유동인구 존재.",
    "성동구": "산업 단지에서 젊은층 문화 공간으로 변화 중. 성수동을 중심으로 트렌디한 상권 발달.",
    "동작구": "주거와 교육 시설이 조화로운 지역. 여의도, 강남 접근성이 양호.",
    "도봉구": "녹지 공간이 풍부한 주거 중심지, 상대적으로 안정적인 인구 구조."
}

# 배경 및 제목 추가 (유지)
st.markdown("<h1 class='header'>🌆 서울 인구 인사이트 대시보드</h1>", unsafe_allow_html=True)
st.markdown("<div class='content'>서울 인구를 Gemini와 함께 재미있게 탐험해 보세요! 클릭 한 번으로 미래 비전까지!</div>", unsafe_allow_html=True)

# ==============================================================================
# 2. 데이터 로드 및 전처리 함수
# ==============================================================================

@st.cache_data
def load_geojson():
    with open("TL_SCCO_SIG.json", encoding="utf-8") as f:
        return json.load(f)

def get_region_name_from_coordinates(lat, lon, geojson):
    point = Point(lon, lat)
    for feature in geojson["features"]:
        polygon = shape(feature["geometry"])
        if polygon.contains(point):
            return feature["properties"]["SIG_KOR_NM"]
    return None

@st.cache_data
def load_population_data(region):
    # male_data와 female_data를 병렬로 로드 (효율성 개선)
    male_data = list(db.population_male.find({"region": region}, {"_id": 0}))
    female_data = list(db.population_female.find({"region": region}, {"_id": 0}))
    
    # 데이터프레임 생성 및 클리닝
    if not male_data: male_df = pd.DataFrame(columns=["year", "population"])
    else:
        male_df = pd.DataFrame(male_data).sort_values("year")
        male_df['population'] = pd.to_numeric(male_df['population'], errors='coerce').fillna(0).astype(int)
        male_df = male_df.dropna(subset=['year'])
    
    if not female_data: female_df = pd.DataFrame(columns=["year", "population"])
    else:
        female_df = pd.DataFrame(female_data).sort_values("year")
        female_df['population'] = pd.to_numeric(female_df['population'], errors='coerce').fillna(0).astype(int)
        female_df = female_df.dropna(subset=['year'])
        
    return male_df, female_df

@st.cache_data
def load_age_data(region):
    try:
        # DB에서 해당 지역의 연령별 데이터 찾기
        age_data_all = list(db.population_by_age.find({"동별(1)": region, "항목": "계"}).sort("_id", -1).limit(1))
        
        if not age_data_all:
             return pd.DataFrame()
        
        age_data_raw = age_data_all[0]
        
        # 가장 최신 연도와 분기 필드 동적으로 찾기 (정확성 개선)
        latest_year_quarter_match = re.search(r"(\d{4}) (\d/\d)\.", ','.join(age_data_raw.keys()))
        if not latest_year_quarter_match:
            st.warning("경고: 연령별 데이터에서 최신 연도/분기 정보를 찾을 수 없습니다.")
            return pd.DataFrame()
            
        latest_prefix = latest_year_quarter_match.group(0) # 예: "2025 1/4."

        age_groups_map = {
            f"{latest_prefix}1": "0-9세", f"{latest_prefix}2": "10-19세", f"{latest_prefix}3": "20-29세",
            f"{latest_prefix}4": "30-39세", f"{latest_prefix}5": "40-49세", f"{latest_prefix}6": "50-59세",
            f"{latest_prefix}7": "60-69세", f"{latest_prefix}8": "70-79세", f"{latest_prefix}9": "80-89세",
            f"{latest_prefix}10": "90-99세", f"{latest_prefix}11": "100세 이상"
        }

        dynamic_age_map = {}
        for k, v in age_data_raw.items():
            if k in age_groups_map:
                dynamic_age_map[age_groups_map[k]] = v
            # 만약 DB 필드가 '2025 1/4.1' 형식이 아니라 '20-29세' 형태로 저장되어 있다면 아래 로직으로 대체
            # elif re.match(r"^\d{1,3}-\d{1,3}세$", k): dynamic_age_map[k] = v

        df = pd.DataFrame(list(dynamic_age_map.items()), columns=['age_group', 'population'])
        df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0).astype(int)
        
        return df

    except Exception as e:
        st.error(f"연령별 데이터 로드 중 오류 발생: {e}")
        return pd.DataFrame()

# ==============================================================================
# 3. 모델링 및 AI 분석 함수
# ==============================================================================

# 예측 함수 (개선: 교차 검증을 통한 신뢰도 높은 R2 계산)
def predict_population(df, end_year=2040):
    if df.empty or len(df) < 5:
        return pd.DataFrame({"year": [], "population": []}), None, 0, 0, 0, 0

    recent_df = df.tail(15).copy()
    if len(recent_df) < 5: # 교차 검증을 위해 최소 데이터 포인트 5개 필요
        recent_df = df.copy() # 최소한의 데이터만 사용

    x = recent_df["year"].values.reshape(-1, 1)
    y = recent_df["population"].values

    best_r2 = -float('inf')
    best_degree = 1
    best_model = None
    
    # 교차 검증을 위한 degree 범위 설정 (데이터 크기에 맞춤)
    max_degree_limit = min(10, len(x) - 1)

    # 교차 검증 (K-Fold)을 통해 최적 차수 및 R2 계산 (신뢰도 향상)
    kf = KFold(n_splits=min(5, len(x)), shuffle=True, random_state=42)
    
    for degree in range(1, max_degree_limit + 1):
        r2_scores = []
        try:
            poly = PolynomialFeatures(degree=degree)
            for train_index, test_index in kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                x_poly_train = poly.fit_transform(x_train)
                model_ = LinearRegression().fit(x_poly_train, y_train)
                
                x_poly_test = poly.transform(x_test)
                y_pred = model_.predict(x_poly_test)
                r2_scores.append(r2_score(y_test, y_pred))
            
            avg_r2 = np.mean(r2_scores)

            # 최적 모델 저장 (전체 데이터 기반 재훈련)
            if avg_r2 > best_r2:
                best_r2 = avg_r2
                best_degree = degree
                
                poly_final = PolynomialFeatures(degree=best_degree)
                x_poly_final = poly_final.fit_transform(x)
                best_model = LinearRegression().fit(x_poly_final, y)
                
        except Exception:
            continue
    
    if best_model is None:
        return pd.DataFrame({"year": [], "population": []}), None, 0, 0, 0, 0
        
    poly_final = PolynomialFeatures(degree=best_degree) # 최종 모델에 사용된 PolynomialFeatures 객체
    
    # 예측 R2 및 MAE 계산 (학술적 신뢰도)
    x_poly_full = poly_final.fit_transform(x)
    y_pred_full = best_model.predict(x_poly_full)
    final_r2 = r2_score(y, y_pred_full)
    final_mae = mean_absolute_error(y, y_pred_full)

    # 미래 예측
    future_years = np.arange(df["year"].max() + 1, end_year + 1).reshape(-1, 1)
    if future_years.size == 0:
        return pd.DataFrame({"year": [], "population": []}), best_model, final_r2, final_mae, best_degree, 0 # MAE 반환

    future_years_poly = poly_final.transform(future_years)
    preds = best_model.predict(future_years_poly)
    preds[preds < 0] = 0
    
    return pd.DataFrame({"year": future_years.flatten(), "population": preds.astype(int)}), best_model, final_r2, final_mae, best_degree, 0

# AI 분석 함수 (유지)
@st.cache_data
def analyze_and_recommend(region, male_df, female_df, age_df, user_query=None, mode='vision'):
    if not GEMINI_CONFIGURED:
        return "<div class='card gemini-highlight'><p class='content warning'>⚠️ Gemini API 키를 찾을 수 없습니다. `.streamlit/secrets.toml` 파일을 확인해주세요.</p></div>"
    if male_df.empty or female_df.empty or len(male_df) < 5:
        return "<div class='card'><p class='content'>💬 <strong>Gemini:</strong> 데이터 부족으로 심층 분석이 어렵습니다. 최소 5년치 이상의 인구 데이터를 확보해 주세요. <span class='grok-style'>- Gemini</span></p></div>"
        
    recent_m = int(male_df.tail(1)["population"].values[0])
    recent_f = int(female_df.tail(1)["population"].values[0])
    recent_year = int(male_df.tail(1)["year"].values[0])
    
    # 인구 변화율 계산
    male_change_rate = 0
    female_change_rate = 0
    trend_m = "변화 없음"
    trend_f = "변화 없음"
    if len(male_df) >= 5:
        male_change_rate = ((male_df.tail(1)["population"].values[0] - male_df.iloc[-5]["population"]) / male_df.iloc[-5]["population"] * 100).round(2)
        trend_m = "증가" if male_change_rate > 0 else "감소" if male_change_rate < 0 else "유지"
    if len(female_df) >= 5:
        female_change_rate = ((female_df.tail(1)["population"].values[0] - female_df.iloc[-5]["population"]) / female_df.iloc[-5]["population"] * 100).round(2)
        trend_f = "증가" if female_change_rate > 0 else "감소" if female_change_rate < 0 else "유지"
        
    # 청년층 인구 비율 계산
    youth_pop = 0
    total_age_pop = age_df['population'].sum()
    youth_age_groups = ['20-29세', '30-39세']
    
    if not age_df.empty:
        youth_df = age_df[age_df['age_group'].isin(youth_age_groups)]
        youth_pop = youth_df['population'].sum() if not youth_df.empty else 0
        youth_ratio = (youth_pop / total_age_pop * 100).round(2) if total_age_pop > 0 else 0
    else:
        youth_ratio = 0
    
    region_traits = REGION_TRAITS.get(region, "다양한 매력을 가진 지역") # 분리된 딕셔너리 사용

    if mode == 'vision':
        prompt_prefix = (
            f"당신은 서울시 인구 변화를 전문적으로 분석하고 미래 비전을 제시하는 AI 분석가 Gemini입니다. "
        )
        prompt_goal = (
            f"미래 발전을 위한 구체적이고 현실적인 3가지 전략적 제안을 해주세요. "
        )
        mode_specific_query = "자유롭게 지역의 미래 비전을 제시해주세요."
    elif mode == 'startup':
        prompt_prefix = (
            f"당신은 서울시 지역별 창업 생태계를 분석하고 혁신적인 아이템을 추천하는 AI 전문가 Gemini입니다. "
        )
        prompt_goal = (
            f"청년 창업가를 위한 구체적이고 실행 가능한 3가지 창업 아이템을 추천해 주세요. "
        )
        mode_specific_query = "청년층을 위한 창업 아이템을 추천해주세요."
    else:
         return "<div class='card gemini-highlight'><p class='content warning'>⚠️ 잘못된 분석 모드입니다. 'vision' 또는 'startup'을 선택해주세요. <span class='grok-style'>- Gemini</span></p></div>"
    
    prompt = (
        prompt_prefix + 
        f"선택된 지역 '{region}'의 인구 데이터를 바탕으로 심층적인 통찰을 제공하고, " +
        prompt_goal +
        f"분석은 객관적이고 모던한 어조로 진행하며, 각 제안은 명확한 근거를 포함해야 합니다.\n\n"
        f"최신 데이터 ({recent_year}년): 남성 {recent_m:,}명, 여성 {recent_f:,}명.\n"
        f"최근 5년간 변화: 남성 {male_change_rate}% ({trend_m}), 여성 {female_change_rate}% ({trend_f}).\n"
        f"청년층(20-39세) 인구: {youth_pop:,}명 (전체 연령 대비 {youth_ratio}%).\n"
        f"지역 특성: {region_traits}.\n"
        f"추가 요청 사항: {user_query if user_query else mode_specific_query}\n\n"
        "답변은 한국어로, 핵심 내용을 명확하게 전달하며, 끝에 '<span class=\"grok-style\">- Gemini</span>'를 추가해 주세요."
    )
        
    try:
        response = model.generate_content(prompt)
        comment = response.text.strip()
        return f"<div class='card gemini-highlight'><p class='content'>💬 <strong>Gemini:</strong> {comment}</p></div>"
    except Exception as e:
        error_message = str(e)
        return f"<div class='card gemini-highlight'><p class='content warning'>⚠️ <strong>Gemini 답변 생성 실패!</strong><br>API 호출 중 오류 발생: {error_message}<br>API 키, 네트워크 연결, 또는 모델 설정을 확인해주세요.</p></div>"

# ==============================================================================
# 4. 시각화 함수
# ==============================================================================

def smooth_curve(x, y):
    # 스무딩 함수 (유지)
    if len(x) < 4: return x, y
    
    x_vals = x.values if isinstance(x, pd.Series) else x
    y_vals = y.values if isinstance(y, pd.Series) else y

    sorted_indices = np.argsort(x_vals)
    x_sorted = x_vals[sorted_indices]
    y_sorted = y_vals[sorted_indices]

    # 중복 X 값 처리: 평균 Y 값 사용
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = np.array([y_sorted[unique_indices[i]:(unique_indices[i+1] if i+1 < len(unique_indices) else len(y_sorted))].mean() for i in range(len(unique_x))])

    if len(unique_x) < 2: return x, y
    if len(unique_x) == 2: k_val = 1
    elif len(unique_x) == 3: k_val = 2
    else: k_val = 3

    x_new = np.linspace(unique_x.min(), unique_x.max(), 300)
    spl = make_interp_spline(unique_x, unique_y, k=k_val)
    return x_new, spl(x_new)

def draw_population_chart(male_df, female_df, male_pred, female_pred, region, male_r2, female_r2, male_degree, female_degree, start_year, end_year, chart_type):
    # 인구 변화 예측 차트 (개선: 예측 신뢰도 표시 개선)
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    fig.patch.set_facecolor('#f9fafb')

    male_df_filtered = male_df[(male_df["year"] >= start_year) & (male_df["year"] <= end_year)].copy()
    female_df_filtered = female_df[(female_df["year"] >= start_year) & (female_df["year"] <= end_year)].copy()
    
    # 예측 시작 연도를 기준으로 실제 데이터와 예측 데이터를 나눔
    last_actual_year = max(male_df["year"].max(), female_df["year"].max()) if not male_df.empty and not female_df.empty else 0

    male_pred_display = male_pred[male_pred["year"] > last_actual_year]
    female_pred_display = female_pred[female_pred["year"] > last_actual_year]

    
    if chart_type == "long_term":
        # 실제 데이터 플로팅
        ax.plot(*smooth_curve(male_df_filtered["year"], male_df_filtered["population"]), label="남성 (실제)", color="#3b82f6", linewidth=2)
        ax.plot(*smooth_curve(female_df_filtered["year"], female_df_filtered["population"]), label="여성 (실제)", color="#ef4444", linewidth=2)
        
        # 예측 데이터 플로팅
        if not male_pred_display.empty: ax.plot(*smooth_curve(male_pred_display["year"], male_pred_display["population"]), label="남성 (예측)", color="#3b82f6", linestyle="--", linewidth=2, alpha=0.7)
        if not female_pred_display.empty: ax.plot(*smooth_curve(female_pred_display["year"], female_pred_display["population"]), label="여성 (예측)", color="#ef4444", linestyle="--", linewidth=2, alpha=0.7)
        
        title = f"{region} 남녀 인구 변화 및 2040년 예측"
        r2_m_display = male_r2
        r2_f_display = female_r2
        best_degree_m = male_degree
        best_degree_f = female_degree
    else: # short_term (2025년 기준)
        # 단기 예측을 위한 2025년 기준 데이터 필터링
        male_df_short_term = male_df[male_df["year"] <= 2025].copy()
        female_df_short_term = female_df[female_df["year"] <= 2025].copy()
        
        male_pred_short, _, male_r2_short, _, male_degree_short, _ = predict_population(male_df_short_term, 2040)
        female_pred_short, _, female_r2_short, _, female_degree_short, _ = predict_population(female_df_short_term, 2040)
        
        # 실제 데이터 플로팅
        ax.plot(*smooth_curve(male_df_short_term["year"], male_df_short_term["population"]), label="남성 (실제 1995-2025)", color="#3b82f6", linewidth=2)
        ax.plot(*smooth_curve(female_df_short_term["year"], female_df_short_term["population"]), label="여성 (실제 1995-2025)", color="#ef4444", linewidth=2)
        
        # 예측 데이터 플로팅
        male_pred_short_display = male_pred_short[male_pred_short["year"] > 2025]
        female_pred_short_display = female_pred_short[female_pred_short["year"] > 2025]
        
        if not male_pred_short_display.empty: ax.plot(*smooth_curve(male_pred_short_display["year"], male_pred_short_display["population"]), label="남성 (예측 2026-2040)", color="#3b82f6", linestyle="--", linewidth=2, alpha=0.7)
        if not female_pred_short_display.empty: ax.plot(*smooth_curve(female_pred_short_display["year"], female_pred_short_display["population"]), label="여성 (예측 2026-2040)", color="#ef4444", linestyle="--", linewidth=2, alpha=0.7)
        
        title = f"{region} 2026-2040년 인구 예측 (1995-{last_actual_year} 기반)" # 제목 수정: 예측 기반 연도를 동적으로
        
        r2_m_display = male_r2_short
        r2_f_display = female_r2_short
        best_degree_m = male_degree_short
        best_degree_f = female_degree_short
        
    # 차트 스타일링 (유지)
    ax.set_title(title, fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    
    # X축 범위 조정 (유지)
    min_x = min(male_df_filtered["year"].min(), female_df_filtered["year"].min()) if not male_df_filtered.empty and not female_df_filtered.empty else start_year
    max_x = max(male_df_filtered["year"].max(), female_df_filtered["year"].max()) if not male_df_filtered.empty and not female_df_filtered.empty else end_year
    
    ax.set_xlim(min_x, max(max_x, end_year))
    ax.set_xticks(range(int(min_x), int(max(max_x, end_year)) + 1, 5))

    st.pyplot(fig)

    # 예측 정확도 표시 (개선: R2 0.7 기준)
    warning = ""
    if r2_m_display < 0.7: warning += "<p class='warning'>⚠️ 남성 예측 정확도(R² < 0.7)로 신뢰도가 낮습니다. 데이터 추가 권장.</p>"
    if r2_f_display < 0.7: warning += "<p class='warning'>⚠️ 여성 예측 정확도(R² < 0.7)로 신뢰도가 낮습니다. 데이터 추가 권장.</p>"
    
    st.markdown(f"""
        <div class='card'>
            <h3 class='subheader'>예측 모델 상세 정보</h3>
            <p class='content'>
                최적 다항 회귀 모델 (차수: 남성 **{best_degree_m}차**, 여성 **{best_degree_f}차**).<br>
                모델 적합도 (교차 검증 평균 R²): 남성 **{r2_m_display:.2f}**, 여성 **{r2_f_display:.2f}** (0.7 이상 권장).
            </p>
            {warning}
        </div>
        """, unsafe_allow_html=True)


def draw_2024_prediction_comparison_chart(male_df, female_df, region):
    # 2024년 예측 vs 실제 비교 차트 (유지)
    if male_df.empty or female_df.empty or 2024 not in male_df['year'].values or 2024 not in female_df['year'].values:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 2024년 실제 데이터가 없어 예측 비교를 할 수 없습니다.</p></div>", unsafe_allow_html=True)
        return
        
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    
    train_df_full = total_df[total_df["year"] <= 2023]
    
    if len(train_df_full) < 5: # 최소 5개년 데이터 필요
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 2023년까지의 데이터가 부족하여 예측 모델을 생성할 수 없습니다. (최소 5개년 데이터 필요)</p></div>", unsafe_allow_html=True)
        return
        
    train_df = train_df_full.tail(15) # 예측에 사용할 데이터 (최근 15년)
    actual_2024_pop = total_df[total_df["year"] == 2024]["total"].iloc[0]
    x_train = train_df["year"].values.reshape(-1, 1)
    y_train = train_df["total"].values
    
    best_r2_comp = -float('inf')
    best_degree_comp = 1
    best_model_comp = None
    best_poly_comp = None
    
    max_degree_comp_limit = min(10, len(x_train) - 1)

    # 교차 검증을 통해 최적 모델 차수 선택
    kf = KFold(n_splits=min(5, len(x_train)), shuffle=True, random_state=42)
    
    for degree in range(1, max_degree_comp_limit + 1):
        r2_scores = []
        try:
            poly_comp = PolynomialFeatures(degree=degree)
            for train_index, test_index in kf.split(x_train):
                # 교차 검증 데이터 준비
                x_sub_train, x_sub_test = x_train[train_index], x_train[test_index]
                y_sub_train, y_sub_test = y_train[train_index], y_train[test_index]
                
                # 모델 학습 및 평가
                x_poly_sub_train = poly_comp.fit_transform(x_sub_train)
                model_comp = LinearRegression().fit(x_poly_sub_train, y_sub_train)
                
                x_poly_sub_test = poly_comp.transform(x_sub_test)
                y_pred = model_comp.predict(x_poly_sub_test)
                r2_scores.append(r2_score(y_sub_test, y_pred))

            avg_r2 = np.mean(r2_scores)
            if avg_r2 > best_r2_comp:
                best_r2_comp = avg_r2
                best_degree_comp = degree
                
                # 최종 모델 재훈련 (전체 훈련 데이터 기반)
                best_poly_comp = PolynomialFeatures(degree=best_degree_comp)
                x_poly_final = best_poly_comp.fit_transform(x_train)
                best_model_comp = LinearRegression().fit(x_poly_final, y_train)

        except Exception:
            continue
            
    if best_model_comp is None or best_poly_comp is None:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 최적 예측 모델을 찾을 수 없습니다.</p></div>", unsafe_allow_html=True)
        return
        
    year_to_predict_poly = best_poly_comp.transform([[2024]])
    predicted_2024_pop = best_model_comp.predict(year_to_predict_poly)[0]
    
    if predicted_2024_pop < 0:
        predicted_2024_pop = 0

    # 오차율 계산: (예측값 - 실제값) / 실제값 * 100
    error_percent = ((predicted_2024_pop - actual_2024_pop) / actual_2024_pop) * 100
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    fig.patch.set_facecolor('#f9fafb')
    
    # 실제 데이터 플롯
    ax.plot(train_df_full["year"], train_df_full["total"], label="총인구 (실제)", color="#10b981", linewidth=2)
    # 2024년 실제값
    ax.plot(2024, actual_2024_pop, 'o', color="#3b82f6", markersize=10, label=f"2024년 실제 인구: {int(actual_2024_pop):,}명")
    # 2024년 예측값
    ax.plot(2024, predicted_2024_pop, 'X', color="#ef4444", markersize=10, label=f"2024년 예측 인구: {int(predicted_2024_pop):,}명")
    
    # 예측 연결선
    last_actual_year = train_df_full["year"].max()
    last_actual_pop = train_df_full[train_df_full["year"] == last_actual_year]["total"].iloc[0]
    
    ax.plot([last_actual_year, 2024], [last_actual_pop, predicted_2024_pop], linestyle="--", color="#ef4444", alpha=0.7)
    
    ax.set_title(f"{region} 2024년 총인구 예측 vs 실제 (오차: {error_percent:.2f}%)", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("총 인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best", frameon=True, facecolor="white")
    st.pyplot(fig)


def draw_age_distribution_chart(df, region):
    # 연령별 인구 분포 차트 (유지)
    if df.empty:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 연령별 인구 데이터가 없습니다.</p></div>", unsafe_allow_html=True)
        return
    
    age_group_order = ["0-9세", "10-19세", "20-29세", "30-39세", "40-49세",
                       "50-59세", "60-69세", "70-79세", "80-89세", "90-99세", "100세 이상"]
    
    age_groups = df.groupby('age_group')['population'].sum().reindex(age_group_order).fillna(0)
    age_groups = age_groups[age_groups > 0]

    if age_groups.empty:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 유효한 연령대별 그룹 데이터가 없어 그래프를 그릴 수 없습니다.</p></div>", unsafe_allow_html=True)
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    fig.patch.set_facecolor('#f9fafb')

    wedges, texts, autotexts = ax.pie(age_groups, labels=age_groups.index, autopct=lambda p: f'{p:.1f}% ({int(p*sum(age_groups)/100):,}명)',
                                       startangle=90, colors=plt.cm.Pastel1.colors, pctdistance=0.85, wedgeprops={'edgecolor': 'white'})
    
    for autotext in autotexts:
        autotext.set_color('gray')
        autotext.set_fontsize(9)
    for text in texts:
        text.set_fontsize(10)

    ax.set_title(f"{region} 최신 연도 연령별 인구 분포", fontsize=16, pad=20, color="#1f2937")
    ax.axis('equal')
    
    plt.tight_layout()
    st.pyplot(fig)

def draw_total_population_chart(male_df, female_df, male_pred, female_pred, region):
    # 총인구 변화 및 예측 차트 (유지)
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    
    total_pred = pd.DataFrame()
    if not male_pred.empty and not female_pred.empty and len(male_pred) == len(female_pred):
        total_pred = pd.DataFrame({"year": male_pred["year"], "total": male_pred["population"] + female_pred["population"]}) 
    
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    fig.patch.set_facecolor('#f9fafb')
    
    ax.plot(*smooth_curve(total_df["year"], total_df["total"]), label="총인구 (실제)", color="#10b981", linewidth=2)
    
    if not total_pred.empty:
        last_actual_year = total_df["year"].max()
        total_pred_display = total_pred[total_pred["year"] > last_actual_year]
        if not total_pred_display.empty:
            ax.plot(*smooth_curve(total_pred_display["year"], total_pred_display["total"]), label="총인구 (예측)", color="#10b981", linestyle="--", linewidth=2, alpha=0.7)
    
    ax.set_title(f"{region} 총인구 변화 및 예측", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("총 인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    st.pyplot(fig)

def draw_population_histogram(male_df, female_df, region):
    # 인구 히스토그램 (유지)
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    
    if total_df.empty:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 인구 데이터가 없습니다. 히스토그램을 그릴 수 없습니다.</p></div>", unsafe_allow_html=True)
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    ax.bar(total_df["year"], total_df["total"], color="#93c5fd", alpha=0.7)
    ax.set_title(f"{region} 연도별 총 인구 히스토그램", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("총 인구 수 (명)", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, linestyle="--", alpha=0.5)
    
    if len(total_df["year"]) > 10:
        ax.set_xticks(total_df["year"].iloc[::(len(total_df["year"]) // 5) or 1])
    else:
        ax.set_xticks(total_df["year"])
    
    st.pyplot(fig)

def draw_growth_rate_chart(male_df, female_df, region):
    # 인구 성장률 차트 (유지)
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    total_df = total_df.sort_values("year")
    
    total_df.reset_index(drop=True, inplace=True)
    
    if len(total_df) < 2:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 인구 성장률을 계산할 데이터가 부족합니다 (최소 2개년 데이터 필요).</p></div>", unsafe_allow_html=True)
        return

    growth_rate = total_df["total"].pct_change() * 100
    
    growth_df = pd.DataFrame({"year": total_df["year"], "growth_rate": growth_rate}).dropna()
    
    if growth_df.empty:
        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 인구 성장률을 계산할 데이터가 부족합니다.</p></div>", unsafe_allow_html=True)
        return

    fig, ax = plt.subplots(figsize=(14, 6), facecolor='#f9fafb')
    ax.plot(growth_df["year"], growth_df["growth_rate"], label="인구 성장률 (%)", color="#f59e0b", linewidth=2, marker='o', markersize=4, linestyle='-')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.set_title(f"{region} 연도별 인구 성장률", fontsize=16, pad=10, color="#1f2937")
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("성장률 (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="upper right", frameon=True, facecolor="white")
    st.pyplot(fig)

def simulate_population(male_df, female_df, growth_rate=0.0):
    # 시뮬레이션 함수 (유지)
    # R2, MAE 등 평가 지표는 시뮬레이션에서는 필요 없으므로 무시
    male_pred, _, _, _, _, _ = predict_population(male_df)
    female_pred, _, _, _, _, _ = predict_population(female_df)
    if not male_pred.empty and not female_pred.empty:
        male_pred["population"] = (male_pred["population"] * (1 + growth_rate)).astype(int)
        female_pred["population"] = (female_pred["population"] * (1 + growth_rate)).astype(int)
        male_pred["population"] = male_pred["population"].apply(lambda x: max(0, x))
        female_pred["population"] = female_pred["population"].apply(lambda x: max(0, x))
    return male_pred, female_pred

def get_region_stats(male_df, female_df):
    # 지역 통계 요약 함수 (유지)
    total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
    total_df["total"] = total_df["population_male"] + total_df["population_female"]
    
    if total_df.empty:
        return {"최대 인구": 0, "최소 인구": 0, "평균 인구": 0}

    return {"최대 인구": int(total_df["total"].max()),
            "최소 인구": int(total_df["total"].min()),
            "평균 인구": int(total_df["total"].mean())}

def download_data(male_df, female_df, male_pred, female_pred, region):
    # 데이터 다운로드 함수 (유지)
    male_pred_df = male_pred if not male_pred.empty else pd.DataFrame(columns=["year", "population"])
    female_pred_df = female_pred if not female_pred.empty else pd.DataFrame(columns=["year", "population"])

    combined_df = pd.concat([male_df.assign(gender="남성", type="실제"),
                             female_df.assign(gender="여성", type="실제"),
                             male_pred_df.assign(gender="남성", type="예측"),
                             female_pred_df.assign(gender="여성", type="예측")])
    combined_df = combined_df.rename(columns={"year": "연도", "population": "인구 수", "gender": "성별", "type": "데이터 유형"})
    output = io.BytesIO()
    combined_df.to_csv(output, index=False, encoding='utf-8-sig')
    return output.getvalue()

# ==============================================================================
# 5. 메인 Streamlit 로직
# ==============================================================================

def main():
    with st.sidebar:
        st.markdown("<h2 class='subheader'>설정 패널</h2>", unsafe_allow_html=True)
        if st.button("새로고침 🔄", key="refresh"):
            st.cache_data.clear()
            st.rerun()
        st.markdown("<div class='info-panel'>팁: 지도를 클릭해 구를 선택하고 Gemini와 대화해 보세요! 📍</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        visualization_type = st.radio(
            "시각화 모드 선택",
            ("히트맵", "인구 추이 라인 차트"),
            index=0,
            key="visualization_radio",
            help="히트맵은 서울시 전체 인구 밀도를 보여주며, 인구 추이 라인 차트는 선택된 지역의 인구 변화를 시각화합니다."
        )
        st.markdown("<hr>", unsafe_allow_html=True)

    # 세션 상태 초기화 (유지)
    if 'last_comment_region' not in st.session_state: st.session_state['last_comment_region'] = None
    if 'ai_comment' not in st.session_state: st.session_state['ai_comment'] = ""
    if 'analysis_mode' not in st.session_state: st.session_state['analysis_mode'] = 'vision'
    if 'selected_region' not in st.session_state or not st.session_state['selected_region']: st.session_state['selected_region'] = "강남구"

    geojson = load_geojson()
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11, tiles="cartodbpositron", zoom_control=True)
    heat_data = []
    all_regions = [f["properties"]["SIG_KOR_NM"] for f in geojson["features"]]
    
    # 히트맵 데이터 준비 (개선: load_population_data 한 번만 호출)
    for region_name in all_regions:
        male_df, female_df = load_population_data(region_name) # 한 번만 호출
        if not male_df.empty and not female_df.empty:
            total_df = male_df.merge(female_df, on="year", suffixes=("_male", "_female"))
            latest_pop_row = total_df.sort_values('year', ascending=False).iloc[0] if not total_df.empty else None
            if latest_pop_row is not None:
                total_pop = int(latest_pop_row['population_male'] + latest_pop_row['population_female'])
                feature = next((f for f in geojson["features"] if f["properties"]["SIG_KOR_NM"] == region_name), None)
                if feature:
                    # Folium을 위한 중앙 좌표 계산
                    centroid = shape(feature["geometry"]).centroid
                    heat_data.append([centroid.y, centroid.x, total_pop])

    if heat_data: HeatMap(heat_data, radius=20, blur=15, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    # GeoJson 레이어 추가 (클릭 이벤트 처리를 위함)
    folium.GeoJson(geojson, name="구", tooltip=folium.GeoJsonTooltip(fields=["SIG_KOR_NM"], aliases=["구 이름"]), style_function=lambda x: {'fillColor': '#42A5F5', 'color': '#283593', 'weight': 1, 'fillOpacity': 0.6}, highlight_function=lambda x: {'fillColor': '#BBDEFB', 'color': '#283593', 'weight': 2}).add_to(m)
    m.add_child(folium.LatLngPopup())

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        if visualization_type == "히트맵":
            st.markdown("<div class='card'><h2 class='subheader'>서울시 인구 밀도 히트맵 🗺️</h2></div>", unsafe_allow_html=True)
            st.markdown("<div class='map-container'>", unsafe_allow_html=True)
            st_map = st_folium(m, width=600, height=600, key="seoul_map")
            st.markdown("</div>", unsafe_allow_html=True)
        elif visualization_type == "인구 추이 라인 차트":
            # 서울시 전체 인구 추이 (유지)
            st.markdown("<div class='card'><h2 class='subheader'>서울시 인구 변화 추이 📈</h2></div>", unsafe_allow_html=True)
            
            # 모든 지역의 데이터를 불러와 합산
            all_male_df = pd.concat([load_population_data(r)[0] for r in all_regions if not load_population_data(r)[0].empty], ignore_index=True)
            all_female_df = pd.concat([load_population_data(r)[1] for r in all_regions if not load_population_data(r)[1].empty], ignore_index=True)
            
            if not all_male_df.empty and not all_female_df.empty:
                all_male_df = all_male_df.groupby('year').sum().reset_index()
                all_female_df = all_female_df.groupby('year').sum().reset_index()
                
                # 예측 모델 재사용을 위해 R2 등은 무시
                male_pred, _, _, _, _, _ = predict_population(all_male_df) 
                female_pred, _, _, _, _, _ = predict_population(all_female_df)
                
                draw_total_population_chart(all_male_df, all_female_df, male_pred, female_pred, "서울시 전체")
            else:
                st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 서울시 전체 인구 데이터를 불러올 수 없습니다.</p></div>", unsafe_allow_html=True)
            st_map = None # 지도 클릭 방지
        
    with col2:
        st.markdown("<div class='card'><h2 class='subheader'>지역 탐색 & Gemini 분석 🔍</h2></div>", unsafe_allow_html=True)
        selected_region_placeholder = st.empty()
        region = st.session_state['selected_region'] # 기본값 '강남구' 또는 클릭된 지역
        
        # 지도 클릭 이벤트 처리 (유지)
        if visualization_type == "히트맵" and st_map and st_map.get("last_clicked"):
            lat, lon = st_map["last_clicked"]["lat"], st_map["last_clicked"]["lng"]
            clicked_region = get_region_name_from_coordinates(lat, lon, geojson)
            if clicked_region and st.session_state.get('selected_region') != clicked_region:
                st.session_state['selected_region'] = clicked_region
                st.session_state['last_comment_region'] = None 
                # st.rerun() # 불필요한 rerunning 방지
                
        # 지역 재설정
        region = st.session_state['selected_region']
        
        selected_region_placeholder.markdown(f"<div class='card'><p class='content'>✅ **{region}** 분석 중... 🚀</p></div>", unsafe_allow_html=True)
        male_df, female_df = load_population_data(region)
        age_df = load_age_data(region)

        if male_df.empty or female_df.empty or len(male_df) < 1:
            st.markdown("<div class='card'><p class='content'>⚠️ 데이터 부족! 최소 1개년 이상의 인구 데이터를 추가하세요.</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><h3 class='subheader'>Gemini의 실시간 코멘트 💡</h3></div>", unsafe_allow_html=True)
            
            # Gemini 분석 모드 라디오 버튼 (유지)
            current_analysis_mode = st.radio(
                "분석 모드 선택",
                ('미래 비전 제시', '청년 창업 아이템 추천'),
                key=f"analysis_mode_{region}",
                index=0 if st.session_state['analysis_mode'] == 'vision' else 1,
                horizontal=True,
                help="미래 비전은 지역의 전반적인 인구 변화 및 발전 방향을 제시하고, 청년 창업 아이템 추천은 청년층 인구 특성을 기반으로 한 창업 아이디어를 제공합니다."
            )
            
            new_mode = 'vision' if current_analysis_mode == '미래 비전 제시' else 'startup'

            # 모드가 바뀌거나 지역이 바뀌면 AI 코멘트 재생성 (유지)
            if (st.session_state['analysis_mode'] != new_mode) or (st.session_state['last_comment_region'] != region):
                st.session_state['analysis_mode'] = new_mode
                st.session_state['last_comment_region'] = region 
                with st.spinner('Gemini가 분석 중... 🤖'):
                    st.session_state['ai_comment'] = analyze_and_recommend(region, male_df, female_df, age_df, mode=st.session_state['analysis_mode'])
            
            comment_placeholder = st.empty()
            comment_placeholder.markdown(st.session_state['ai_comment'], unsafe_allow_html=True)
            
            user_query_placeholder = "자유롭게 추가 질문을 입력해 주세요."
            if st.session_state['analysis_mode'] == 'startup':
                user_query_placeholder = "예: 이 지역의 고유한 문화 요소를 활용한 창업 아이템은 무엇이 있을까요?"

            user_query = st.text_input("Gemini에게 더 물어보기 🗣️", key=f"query_{region}_{st.session_state['analysis_mode']}", placeholder=user_query_placeholder)
            
            if st.button("추가 질문 보내기 ❓", key=f"ask_{region}_{st.session_state['analysis_mode']}"):
                if user_query:
                    with st.spinner('Gemini가 답변 중... 💭'):
                        st.session_state['ai_comment'] = analyze_and_recommend(region, male_df, female_df, age_df, user_query=user_query, mode=st.session_state['analysis_mode'])
                    st.rerun() # 답변 반영을 위해 rerun
                else:
                    st.warning("질문을 입력해 주세요!")
        
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if 'selected_region' in st.session_state and st.session_state['selected_region']:
        region = st.session_state['selected_region']
        male_df, female_df = load_population_data(region)
        
        if not male_df.empty and not female_df.empty and len(male_df) >= 1:
            # 예측 모델은 한 번만 계산하여 모든 탭에서 재활용
            # R2, MAE, degree는 교차 검증 기반으로 계산되어 높은 신뢰도를 가짐
            male_pred, _, male_r2, male_mae, male_degree, _ = predict_population(male_df)
            female_pred, _, female_r2, female_mae, female_degree, _ = predict_population(female_df)
            
            if len(male_df) >= 2: # 탭 구성은 데이터가 충분할 때만
                tab1, tab2, tab3, tab4 = st.tabs(["📊 인구 통계", "📈 예측 분석", "🎮 시뮬레이션", "🔍 상세 그래프"])

                with tab1:
                    col_stats, col_hist = st.columns([1, 1])
                    with col_stats:
                        stats = get_region_stats(male_df, female_df)
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 통계 요약 📝</h3></div>", unsafe_allow_html=True)
                        st.table(pd.DataFrame({
                            "지표": ["최대 인구", "최소 인구", "평균 인구"],
                            "값": [f"{v:,}" for v in stats.values()]
                        }))
                    with col_hist:
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 총 인구 히스토그램 📊</h3></div>", unsafe_allow_html=True)
                        draw_population_histogram(male_df, female_df, region)
                    
                with tab2:
                    if not (2024 in male_df['year'].values and 2024 in female_df['year'].values):
                        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 2024년 실제 데이터가 없어 '2024년 예측 vs 실제' 비교 그래프를 표시할 수 없습니다.</p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 2024년 예측 vs 실제 📈</h3></div>", unsafe_allow_html=True)
                        draw_2024_prediction_comparison_chart(male_df, female_df, region)
                    
                    years = list(male_df["year"].dropna().unique())
                    if len(years) < 2:
                        st.markdown("<div class='card'><p class='content text-yellow-600'>⚠️ 인구 예측 그래프를 그릴 데이터가 부족합니다 (최소 2개년 데이터 필요).</p></div>", unsafe_allow_html=True)
                    else:
                        min_year, max_year = int(min(years)), int(max(years))
                        slider_max_year = max(max_year, 2040) 

                        start_year, end_year = st.slider("연도 범위 선택 ⏳", min_year, slider_max_year, (min_year, slider_max_year), key="tab2_slider")
                        
                        chart_type = st.selectbox("그래프 타입 선택 📊", ["long_term", "short_term"], index=0, help="long_term: 전체 추세 (모든 실제 데이터 기반 예측) | short_term: 단기 예측 (2025년까지의 실제 데이터 기반 예측)", key="tab2_chart_type")
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 남녀 인구 예측 👥</h3></div>", unsafe_allow_html=True)
                        
                        draw_population_chart(male_df, female_df, male_pred, female_pred, region, male_r2, female_r2, male_degree, female_degree, start_year, end_year, chart_type)
                    
                with tab3:
                    st.markdown(f"<div class='card'><h3 class='subheader'>{region} 인구 시뮬레이션 🎮</h3></div>", unsafe_allow_html=True)
                    growth_rate = st.slider("성장률 조정 (%) 🔄", -5.0, 5.0, 0.0, 0.1, key="growth_slider_tab3")
                    male_pred_sim, female_pred_sim = simulate_population(male_df, female_df, growth_rate / 100)
                    draw_total_population_chart(male_df, female_df, male_pred_sim, female_pred_sim, region)

                with tab4:
                    col_age, col_growth = st.columns([1, 1])
                    with col_age:
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 최신 연도 연령별 인구 분포 👶👵</h3></div>", unsafe_allow_html=True)
                        age_df = load_age_data(region)
                        draw_age_distribution_chart(age_df, region)
                    with col_growth:
                        st.markdown(f"<div class='card'><h3 class='subheader'>{region} 성장률 추이 📈</h3></div>", unsafe_allow_html=True)
                        draw_growth_rate_chart(male_df, female_df, region)

                # 데이터 다운로드 버튼 (유지)
                csv_data = download_data(male_df, female_df, male_pred, female_pred, region)
                st.download_button(label="데이터 다운로드 📥 (CSV)", data=csv_data, file_name=f"{region}_population.csv", mime="text/csv")
            else:
                st.markdown("<div class='card'><p class='content'>ℹ️ 선택된 지역의 인구 데이터를 분석하기에 데이터가 충분하지 않습니다. (최소 2개년 데이터 필요) 🌟</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><p class='content'>ℹ️ 지도를 클릭해 구를 선택하고 Gemini와 대화 시작! 🌟</p></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()