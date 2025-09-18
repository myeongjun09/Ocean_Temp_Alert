# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
from io import StringIO
from datetime import datetime

# 폰트 적용 (Pretendard, 없으면 무시)
plt.rcParams['font.family'] = 'Pretendard'

# 페이지 설정
st.set_page_config(page_title="해수온 상승 대시보드", layout="wide")
st.title("🌊 바다의 온도 경고음: 해수온 상승과 지속 가능한 해결책")

# =========================
# 공개 데이터 대시보드
# =========================
st.header("📈 공개 데이터 기반 해수온 상승 분석")

@st.cache_data
def load_public_data():
    try:
        url = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/2023/AVHRR_OI_v2.1_20230101.csv"
        r = requests.get(url)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] <= pd.Timestamp(datetime.now().date())]  
        df = df[['date', 'value']].drop_duplicates()
        return df
    except:
        st.warning("공개 데이터 로드 실패, 예시 데이터로 대체합니다.")
        dates = pd.date_range(start="2023-01-01", periods=12, freq='M')
        values = np.linspace(26, 28, 12)
        return pd.DataFrame({'date': dates, 'value': values})

public_df = load_public_data()

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=public_df, x='date', y='value', marker='o', ax=ax)
ax.set_title("공개 데이터 기반 월별 해수온 변화", fontsize=14)
ax.set_xlabel("날짜")
ax.set_ylabel("해수온 (℃)")
st.pyplot(fig)

st.download_button(
    label="📥 공개 데이터 다운로드",
    data=public_df.to_csv(index=False),
    file_name='public_sea_temp.csv',
    mime='text/csv'
)

# =========================
# 사용자 입력 데이터
# =========================
st.header("📝 사용자 입력 데이터 기반 해수온 및 해양 생태계 영향 분석")

@st.cache_data
def load_user_data():
    dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
    values = [26.1, 26.5, 27.0, 27.2, 27.8,
              28.0, 28.3, 28.5, 28.6, 28.9, 29.0, 29.2]
    groups = ["서해"]*6 + ["남해"]*6
    coral_bleaching = [5, 6, 7, 8, 10, 12, 14, 15, 16, 18, 19, 20]
    fish_migration_change = [2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9]
    flood_risk = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    df = pd.DataFrame({
        'date': dates,
        '해수온': values,
        '지역': groups,
        '산호초 백화(%)': coral_bleaching,
        '어류 이동 변화(%)': fish_migration_change,
        '침수 위험 지수': flood_risk
    })
    df['date'] = pd.to_datetime(df['date'])
    df['연도'] = df['date'].dt.year
    df = df[df['date'] <= pd.Timestamp(datetime.now().date())]
    return df

user_df = load_user_data()

# 사이드바 필터
st.sidebar.header("⚙️ 데이터 필터")
selected_region = st.sidebar.multiselect(
    "지역 선택", options=user_df['지역'].unique(), default=user_df['지역'].unique()
)
selected_year = st.sidebar.selectbox(
    "연도 선택", options=sorted(user_df['연도'].unique()), index=len(user_df['연도'].unique())-1
)

df_filtered = user_df[(user_df['지역'].isin(selected_region)) & (user_df['연도'] == selected_year)]

# 해수온 라인 차트
st.subheader(f"📊 {selected_year}년 월별 해수온 변화")
fig_temp = px.line(df_filtered, x='date', y='해수온', color='지역',
                   labels={'date': '날짜', '해수온': '해수온 (℃)', '지역': '지역'})
st.plotly_chart(fig_temp, use_container_width=True)

# 산호초, 어류, 침수 위험 그래프
fig_ecosystem = go.Figure()
for region in df_filtered['지역'].unique():
    df_r = df_filtered[df_filtered['지역'] == region]
    fig_ecosystem.add_trace(
        go.Bar(x=df_r['date'], y=df_r['산호초 백화(%)'], name=f"{region} 산호초 백화"))
    fig_ecosystem.add_trace(
        go.Bar(x=df_r['date'], y=df_r['어류 이동 변화(%)'], name=f"{region} 어류 이동"))
    fig_ecosystem.add_trace(
        go.Bar(x=df_r['date'], y=df_r['침수 위험 지수'], name=f"{region} 침수 위험"))

fig_ecosystem.update_layout(barmode='group', title=f"{selected_year}년 해양 생태계 영향 지표")
st.plotly_chart(fig_ecosystem, use_container_width=True)

st.download_button(
    label="📥 사용자 데이터 다운로드",
    data=df_filtered.to_csv(index=False),
    file_name=f'user_sea_temp_ecosystem_{selected_year}.csv',
    mime='text/csv'
)

# =========================
# 이미지 예시 섹션
# =========================
st.header("🌐 해수온 상승 관련 시각 자료")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/4e/Coral_bleaching_example.jpg",
        caption="산호초 백화현상 (출처: 위키미디어)",
        use_container_width=True
    )

with col2:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/f0/Global_Warming_Map.jpg",
        caption="지구 평균 해수온 변화 지도 (출처: 위키미디어)",
        use_container_width=True
    )

# =========================
# 지도 시각화
# =========================
st.header("🗺 해안 도시 침수 위험 지도 (예시)")

map_df = pd.DataFrame({
    '위도': [37.56, 35.17, 34.75],
    '경도': [126.97, 129.07, 127.07],
    '도시': ['서울', '부산', '대구'],
    '침수 위험 지수': [3, 5, 2]
})

st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=36.5,
        longitude=128,
        zoom=5,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            'ColumnLayer',
            data=map_df,
            get_position='[경도, 위도]',
            get_elevation='침수 위험 지수',
            elevation_scale=1000,
            radius=5000,
            get_fill_color='[255, 0, 0, 140]',
            pickable=True,
            auto_highlight=True
        )
    ],
    tooltip={"text": "{도시}\n침수 위험 지수: {침수 위험 지수}"}
))

# =========================
# 결론
# =========================
st.header("💡 결론 및 제언")
st.markdown("""
- 해수온 상승으로 산호초 백화, 어류 이동 경로 변화, 해안 도시 침수 등 문제가 발생합니다.  
- 원인은 인간 활동에 의한 온실가스 배출이며, 국제적 정책 대응과 개인 실천이 필요합니다.  
- 학생 개개인의 작은 행동(일회용품 줄이기, 에너지 절약 등)도 장기적으로 큰 효과를 발휘할 수 있습니다.  
""")
