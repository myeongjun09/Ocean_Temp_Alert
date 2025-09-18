import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Matplotlib 한글 폰트 설정
# 시스템에 나눔고딕이 설치되어 있지 않다면 설치해야 합니다.
# 아래 코드는 로컬 환경에서 나눔고딕 폰트가 설치되어 있다고 가정합니다.
try:
    font_path = fm.findfont(fm.FontProperties(family='NanumGothic'))
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 나눔고딕 폰트가 없을 경우 다른 폰트 사용
    # 예: Windows 환경의 맑은 고딕
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 페이지 기본 설정
# -----------------------
st.set_page_config(
    page_title="바다의 온도 경고음",
    page_icon="🌊",
    layout="wide"
)

# -----------------------
# 제목
# -----------------------
st.title("🌊 바다의 온도 경고음: 해수온 상승과 지속 가능한 해결책")
st.markdown("""
이 앱은 해수온 상승의 원인과 영향을 분석하고, 데이터를 시각화하여 보여줍니다. 
또한 사용자가 직접 **연도 범위와 지역**을 선택해 변화를 확인할 수 있습니다.
""")

# -----------------------
# 사이드바: 지역, 연도, 그래프 선택
# -----------------------
st.sidebar.header("🌍 지역 선택")
regions = ["전 세계", "한국 연안", "태평양", "대서양", "인도양", "북극해", "남극해"]
selected_region = st.sidebar.selectbox("지역을 선택하세요", regions)

st.sidebar.header("📅 연도 선택")
years = np.arange(2000, 2026)
year_range = st.sidebar.slider(
    "연도 범위를 선택하세요",
    min_value=int(years.min()),
    max_value=int(years.max()),
    value=(2010, 2020)
)

st.sidebar.header("📈 그래프 설정")
chart_type = st.sidebar.selectbox(
    "그래프 종류를 선택하세요",
    ["선 그래프", "막대 그래프"]
)

st.sidebar.header("📋 설문조사 선택")
survey_options = st.sidebar.multiselect(
    "보고 싶은 설문 항목 선택",
    ["해수온 상승의 원인", "영향 인식 정도", "가장 효과적인 해결 방법", "일상 실천 방안"],
    default=["해수온 상승의 원인", "가장 효과적인 해결 방법"]
)

# -----------------------
# 예시 데이터 생성
# -----------------------
region_temp_offset = {
    "전 세계": 0,
    "한국 연안": 0.2,
    "태평양": 0.3,
    "대서양": 0.1,
    "인도양": 0.25,
    "북극해": 1.0,
    "남극해": 0.8
}
temperatures = 15 + region_temp_offset[selected_region] + 0.03 * (years - 2000) + np.random.normal(0, 0.1, len(years))
df_temp = pd.DataFrame({"연도": years, "평균 해수온도(°C)": temperatures})
filtered_df = df_temp[(df_temp["연도"] >= year_range[0]) & (df_temp["연도"] <= year_range[1])]

# -----------------------
# 설문조사 데이터
# -----------------------
survey_q1 = pd.DataFrame({
    "원인": ["온실가스 배출", "산업 폐기물", "해양 쓰레기", "기타"],
    "응답 수": [2, 1, 0, 1]
})
survey_q2 = pd.DataFrame({
    "영향 인식 정도": ["1", "2", "3", "4", "5"],
    "응답 수": [0, 0, 0, 2, 1]
})
survey_q3 = pd.DataFrame({
    "해결 방법": ["온실가스 배출량 줄이기", "해양 보호구역 설치", "자연 기반 해법 활용", "기타"],
    "응답 수": [2, 1, 0, 1]
})
survey_q4 = pd.DataFrame({
    "일상 실천 방법": ["에너지 절약", "일회용품 줄이기", "대중교통/자전거 이용", "재활용 실천", "기타"],
    "응답 수": [0, 2, 3, 0, 1]
})
survey_general = pd.DataFrame({
    "응답": ["매우 심각하다", "심각하다", "보통이다", "별로 심각하지 않다"],
    "비율(%)": [40, 35, 20, 5]
})

# -----------------------
# 탭 구성: 서론, 본론, 결론
# -----------------------
tabs = st.tabs(["서론", "본론", "결론"])

# -----------------------
# 서론
# -----------------------
with tabs[0]:
    st.header("서론")
    st.markdown("""
    지구 온난화는 전 세계적으로 중요한 환경 문제로 대두되고 있으며, 
    특히 해수온 상승은 기후 변화의 가장 뚜렷한 징후 중 하나입니다. 
    해수온 상승은 해양 생태계, 기상 패턴, 인류 사회 전반에 걸쳐 심각한 영향을 미치고 있습니다. 
    
    본 보고서에서는 해수온 상승의 원인과 영향, 
    그리고 지속 가능한 해결 방안을 학생 설문조사 결과와 데이터 시각화를 바탕으로 분석하고자 합니다.
    """)

# -----------------------
# 본론
# -----------------------
with tabs[1]:
    st.header("본론")

    # --- 1. 해수온 추세 ---
    st.subheader(f"📊 {selected_region} 해수온 상승 추세")
    fig, ax = plt.subplots(figsize=(10, 5))
    if chart_type == "선 그래프":
        ax.plot(filtered_df["연도"], filtered_df["평균 해수온도(°C)"], marker="o", linestyle="-")
    elif chart_type == "막대 그래프":
        ax.bar(filtered_df["연도"], filtered_df["평균 해수온도(°C)"])
    
    ax.set_xlabel("연도")
    ax.set_ylabel("평균 해수온도 (°C)")
    ax.set_title(f"{selected_region}: {year_range[0]}년 ~ {year_range[1]}년 해수온 추세")
    ax.tick_params(axis='x', rotation=0)
    st.pyplot(fig)
    st.dataframe(filtered_df, use_container_width=True)

    # --- 2. 해수온 상승 영향 ---
    st.subheader("🌍 해수온 상승의 영향")
    impact_options = st.multiselect(
        "관심 있는 영향을 선택하세요",
        ["산호초 백화현상", "어류 이동 경로 변화", "해안 도시 침수", "극지방 빙하 감소"],
        default=["산호초 백화현상", "어류 이동 경로 변화"]
    )
    col1, col2 = st.columns(2)
    
    image_paths = {
        "산호초 백화현상": "https://upload.wikimedia.org/wikipedia/commons/0/0d/Coral_bleaching.jpg",
        "어류 이동 경로 변화": "https://upload.wikimedia.org/wikipedia/commons/4/49/Fish_school_in_Palau.jpg",
        "해안 도시 침수": "https://upload.wikimedia.org/wikipedia/commons/3/3f/Miami_flooding.jpg",
        "극지방 빙하 감소": "https://upload.wikimedia.org/wikipedia/commons/f/f3/Melting_Glacier.jpg"
    }

    if "산호초 백화현상" in impact_options:
        with col1:
            st.markdown("#### 🪸 산호초 백화현상")
            st.image(image_paths["산호초 백화현상"], caption="산호초 백화현상", use_container_width=True)
            st.write("해수온 상승으로 인해 산호가 스트레스를 받아 백화현상이 발생하며, 이는 해양 생물 다양성 감소로 이어집니다.")
    if "어류 이동 경로 변화" in impact_options:
        with col2:
            st.markdown("#### 🐟 어류 이동 경로 변화")
            st.image(image_paths["어류 이동 경로 변화"], caption="어류 무리 이동", use_container_width=True)
            st.write("어류는 적정 수온을 찾아 이동하는데, 수온 상승으로 기존의 어획 지역이 변화하여 어업 생산량에 영향을 줍니다.")
    if "해안 도시 침수" in impact_options:
        st.markdown("#### 🌆 해안 도시 침수")
        st.image(image_paths["해안 도시 침수"], caption="해수면 상승으로 인한 해안 도시 침수", use_container_width=True)
        st.write("해수온 상승은 해수면 상승으로 이어져, 해안 도시의 침수 위험을 증가시킵니다.")
    if "극지방 빙하 감소" in impact_options:
        st.markdown("#### 🧊 극지방 빙하 감소")
        st.image(image_paths["극지방 빙하 감소"], caption="빙하 감소", use_container_width=True)
        st.write("극지방의 빙하가 빠르게 녹으면서 전 세계 해수면 상승을 가속화합니다.")

    # --- 3. 설문조사 ---
    st.subheader("📋 미림마이스터고 학생 설문조사 결과")
    col1, col2 = st.columns(2)

    if "해수온 상승의 원인" in survey_options:
        with col1:
            st.markdown("1️⃣ 해수온 상승의 원인")
            fig, ax = plt.subplots()
            ax.bar(survey_q1["원인"], survey_q1["응답 수"])
            ax.set_title("해수온 상승의 원인")
            ax.set_xticklabels(survey_q1["원인"], rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

    if "영향 인식 정도" in survey_options:
        with col1:
            st.markdown("2️⃣ 영향 인식 정도")
            fig, ax = plt.subplots()
            ax.bar(survey_q2["영향 인식 정도"], survey_q2["응답 수"])
            ax.set_title("영향 인식 정도")
            plt.tight_layout()
            st.pyplot(fig)

    if "가장 효과적인 해결 방법" in survey_options:
        with col2:
            st.markdown("3️⃣ 가장 효과적인 해결 방법")
            fig, ax = plt.subplots()
            ax.bar(survey_q3["해결 방법"], survey_q3["응답 수"])
            ax.set_title("가장 효과적인 해결 방법")
            ax.set_xticklabels(survey_q3["해결 방법"], rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)

    if "일상 실천 방안" in survey_options:
        with col2:
            st.markdown("4️⃣ 일상 실천 방안")
            fig, ax = plt.subplots()
            ax.bar(survey_q4["일상 실천 방법"], survey_q4["응답 수"])
            ax.set_title("일상 실천 방안")
            ax.set_xticklabels(survey_q4["일상 실천 방법"], rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            
    st.markdown("**전체 학생 인식 정도**")
    fig, ax = plt.subplots()
    ax.bar(survey_general["응답"], survey_general["비율(%)"])
    ax.set_title("전체 학생 인식 정도")
    plt.tight_layout()
    st.pyplot(fig)

# -----------------------
# 결론
# -----------------------
with tabs[2]:
    st.header("결론 및 제언")
    st.markdown("""
- 학생들은 해수온 상승의 **주요 원인으로 온실가스 배출**을 꼽았습니다. 
- 해수온 상승의 **영향 인식 수준은 평균적으로 높은 편(4~5점)**으로 나타났습니다. 
- 해결 방법으로는 **온실가스 감축과 해양 보호구역 설치**가 주요하게 선택되었습니다. 
- 일상에서 실천 가능한 방법으로는 **대중교통/자전거 이용, 일회용품 줄이기**가 많이 언급되었습니다. 

따라서, **개인적인 실천과 정책적 노력의 병행**이 필요합니다. 
특히 교육을 통한 인식 제고와 실천 습관 형성이 중요하며, 
장기적으로는 국제적 협력과 지속 가능한 해양 관리 정책이 뒷받침되어야 합니다.
""")

# 추가: 결론에서 실천 방안 투표
solution = st.radio(
    "당신이 가장 실천할 수 있다고 생각하는 방법은?",
    ["일회용품 줄이기", "대중교통 이용", "재활용 강화", "친환경 제품 구매"]
)
st.success(f"👍 선택하신 실천 방안: {solution}")
st.info("작은 행동이 모여 큰 변화를 만듭니다. 우리 모두가 바다의 목소리에 귀 기울여야 할 때입니다.")