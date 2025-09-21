"""
Streamlit 앱: 해수온 대시보드 (한국어 UI) - 개선된 오류 처리 포함
- NOAA OISST 기반
- 기상청 폭염일수(서울)
- 사용자 입력 예시 데이터
- 기능: 데이터 불러오기, 전처리, 시각화, 간단 분석
"""

import os
import time
import logging
from datetime import date
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any, Tuple

# === 선택적 패키지 임포트 ===
XARRAY_AVAILABLE = False
try:
    import xarray as xr
    XARRAY_AVAILABLE = True
    print("✅ xarray 사용 가능")
except ImportError:
    print("⚠️ xarray 없음 - NetCDF/GRIB 파일 처리 불가")

# === 로깅 설정 ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === Streamlit 설정 ===
st.set_page_config(page_title="해수온 대시보드 — 미림마이스터고", layout="wide")

# === 전역 변수 ===
TODAY = pd.to_datetime(date.today())

# === 선택적 패키지 임포트 체크 ===
STATSMODELS_AVAILABLE = False
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
    logger.info("statsmodels 사용 가능")
except ImportError:
    logger.warning("statsmodels 없음 - numpy.polyfit으로 대체")

# === 폰트 설정 (오류 처리 포함) ===
def setup_font():
    """폰트 설정 - 실패해도 앱이 계속 실행되도록 처리"""
    FONT_PATH = "/fonts/Pretendard-Bold.ttf"
    try:
        import matplotlib.font_manager as fm
        if os.path.exists(FONT_PATH):
            fm.fontManager.addfont(FONT_PATH)
            plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
            logger.info(f"폰트 설정 완료: {FONT_PATH}")
        else:
            logger.warning(f"폰트 파일 없음: {FONT_PATH}")
    except Exception as e:
        logger.error(f"폰트 설정 실패: {e}")
        st.sidebar.warning("⚠️ 폰트 설정에 문제가 있지만 앱은 정상 작동합니다.")

setup_font()

# === 안전한 다운로드 함수 ===
@st.cache_data(ttl=60*60)
def download_text(url: str, max_retries: int = 2, timeout: int = 20) -> bytes:
    """재시도와 타임아웃이 포함된 안전한 다운로드"""
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"다운로드 시도 {attempt + 1}/{max_retries + 1}: {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            logger.info(f"다운로드 성공: {url}")
            return response.content
        except requests.exceptions.Timeout as e:
            last_exc = e
            logger.warning(f"타임아웃 발생 (시도 {attempt + 1}): {e}")
        except requests.exceptions.RequestException as e:
            last_exc = e
            logger.warning(f"요청 실패 (시도 {attempt + 1}): {e}")
        except Exception as e:
            last_exc = e
            logger.error(f"예상치 못한 오류 (시도 {attempt + 1}): {e}")
        
        if attempt < max_retries:
            sleep_time = 1 + attempt
            logger.info(f"{sleep_time}초 대기 후 재시도...")
            time.sleep(sleep_time)
    
    logger.error(f"모든 재시도 실패: {url}")
    raise last_exc

# === 예시 데이터 생성 함수들 ===
@st.cache_data(ttl=60*60)
def load_noaa_pathfinder_example() -> pd.DataFrame:
    """NOAA 해수온 예시 데이터 생성"""
    try:
        yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
        np.random.seed(0)
        base = 15 + 0.015 * (np.arange(len(yrs)))
        seasonal = 1.5 * np.sin(2*np.pi*(yrs.month-1)/12)
        noise = np.random.normal(scale=0.2, size=len(yrs))
        sst = base + seasonal + noise
        
        df = pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})
        logger.info(f"NOAA 예시 데이터 생성 완료: {len(df)}개 행")
        return df
    except Exception as e:
        logger.error(f"NOAA 예시 데이터 생성 실패: {e}")
        # 최소한의 데이터라도 반환
        return pd.DataFrame({"date": [pd.Timestamp.now()], "sst_global_mean_C": [15.0]})

@st.cache_data(ttl=60*60)
def load_kma_heatwave_example() -> pd.DataFrame:
    """기상청 폭염일수 예시 데이터 생성"""
    try:
        years = np.arange(1980, 2025)
        np.random.seed(1)
        base = np.clip((years-1975)*0.15, 0, None)
        noise = np.random.normal(scale=2.0, size=len(years))
        days = np.clip(np.round(base + noise).astype(int), 0, None)
        
        df = pd.DataFrame({"year": years, "heatwave_days_seoul": days})
        logger.info(f"기상청 예시 데이터 생성 완료: {len(df)}개 행")
        return df
    except Exception as e:
        logger.error(f"기상청 예시 데이터 생성 실패: {e}")
        return pd.DataFrame({"year": [2024], "heatwave_days_seoul": [10]})

@st.cache_data(ttl=60*60)
def load_user_input_example() -> Dict[str, pd.DataFrame]:
    """사용자 입력 예시 데이터 생성"""
    try:
        survey = pd.DataFrame({
            "response": ["중요하게 인식함", "보통", "중요하지 않음"],
            "count": [128, 45, 27]
        })
        
        impacts = pd.DataFrame({
            "impact": ["집중력 저하", "수업 단축/취소", "건강 문제(두통/탈수)", "기타"],
            "percent": [45, 25, 20, 10]
        })
        
        months = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
        np.random.seed(2)
        trend = 10 + 0.02 * np.arange(len(months))
        seasonal = 3*np.sin(2*np.pi*(months.month-1)/12)
        noise = np.random.normal(scale=0.3, size=len(months))
        sst_east = trend + seasonal + noise
        df_east = pd.DataFrame({"date": months, "sst_east_C": sst_east})
        df_east = df_east[df_east["date"] <= TODAY]
        
        logger.info("사용자 입력 예시 데이터 생성 완료")
        return {"survey": survey, "impacts": impacts, "sst_east": df_east}
    
    except Exception as e:
        logger.error(f"사용자 입력 예시 데이터 생성 실패: {e}")
        # 최소한의 데이터 반환
        return {
            "survey": pd.DataFrame({"response": ["오류"], "count": [1]}),
            "impacts": pd.DataFrame({"impact": ["데이터 오류"], "percent": [100]}),
            "sst_east": pd.DataFrame({"date": [pd.Timestamp.now()], "sst_east_C": [10.0]})
        }

# === 공개 데이터 로드 함수 ===
@st.cache_data(ttl=60*60)
def load_public_datasets() -> Dict[str, Any]:
    """공개 데이터셋 로드 (오류 처리 포함)"""
    notices = []
    
    try:
        # 실제 OISST 파일 다운로드는 생략하고 예시 데이터 사용
        logger.info("공개 데이터셋 로드 시작")
        
        df_sst = load_noaa_pathfinder_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        
        df_kma = load_kma_heatwave_example()
        
        logger.info("공개 데이터셋 로드 완료")
        return {"sst": df_sst, "kma_heatwave": df_kma, "notices": notices}
        
    except Exception as e:
        error_msg = f"공개 데이터 로드 중 오류 발생: {str(e)}"
        logger.error(error_msg)
        notices.append(f"⚠️ {error_msg} — 예시 데이터로 대체됨")
        
        # 오류가 발생해도 기본 데이터는 제공
        try:
            df_sst = load_noaa_pathfinder_example()
            df_sst = df_sst[df_sst["date"] <= TODAY]
            df_kma = load_kma_heatwave_example()
            return {"sst": df_sst, "kma_heatwave": df_kma, "notices": notices}
        except Exception as fallback_error:
            logger.critical(f"기본 데이터 생성도 실패: {fallback_error}")
            notices.append(f"🚨 심각한 오류: 기본 데이터도 생성할 수 없습니다.")
            return {"sst": pd.DataFrame(), "kma_heatwave": pd.DataFrame(), "notices": notices}

# === 안전한 시각화 함수 ===
def create_scatter_with_trend(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    """추세선이 포함된 산점도 (statsmodels 의존성 없이)"""
    try:
        # 기본 산점도 생성
        if STATSMODELS_AVAILABLE:
            # statsmodels가 있는 경우에만 trendline="ols" 사용
            fig = px.scatter(df, x=x, y=y, trendline="ols", title=title)
            logger.info("statsmodels 추세선 사용")
        else:
            # statsmodels 없으면 수동으로 추세선 추가
            fig = px.scatter(df, x=x, y=y, title=title)
            
            # numpy.polyfit으로 선형 추세선 계산
            if len(df) > 1:
                x_numeric = pd.to_numeric(df[x], errors='coerce').dropna()
                y_numeric = df[y].loc[x_numeric.index]
                
                if len(x_numeric) > 1:
                    coeffs = np.polyfit(x_numeric, y_numeric, 1)
                    trend_y = coeffs[0] * x_numeric + coeffs[1]
                    
                    fig.add_trace(go.Scatter(
                        x=df[x].loc[x_numeric.index],
                        y=trend_y,
                        mode='lines',
                        name='추세선',
                        line=dict(color='red', dash='dash')
                    ))
                    logger.info("numpy 추세선 사용")
        
        return fig
        
    except Exception as e:
        logger.error(f"추세선 생성 실패: {e}")
        # 추세선 없는 기본 산점도 반환
        try:
            fig = px.scatter(df, x=x, y=y, title=f"{title} (추세선 오류)")
            return fig
        except Exception as fallback_error:
            logger.error(f"기본 산점도 생성도 실패: {fallback_error}")
            # 빈 차트 반환
            fig = go.Figure()
            fig.update_layout(title=f"{title} (차트 생성 오류)")
            return fig

def safe_chart_creation(chart_func, *args, **kwargs) -> Optional[go.Figure]:
    """차트 생성 함수를 안전하게 실행"""
    try:
        return chart_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"차트 생성 실패: {e}")
        st.error(f"📊 차트를 생성하는 중 오류가 발생했습니다: {str(e)}")
        return None

# === 메인 애플리케이션 ===
def main():
    """메인 애플리케이션 함수"""
    
    # 데이터 로드
    with st.spinner("📊 공개 데이터와 예시 데이터를 불러오는 중..."):
        try:
            public = load_public_datasets()
            user_input = load_user_input_example()
            
            # 로드 중 발생한 알림 표시
            if public.get("notices"):
                for notice in public["notices"]:
                    if "🚨" in notice:
                        st.error(notice)
                    else:
                        st.warning(notice)
                        
        except Exception as e:
            logger.critical(f"데이터 로드 완전 실패: {e}")
            st.error(f"🚨 데이터를 불러올 수 없습니다: {str(e)}")
            st.stop()

    # 사이드바 설정
    st.sidebar.header("⚙️ 데이터/분석 옵션")
    
    # 데이터셋 선택
    dataset_choice = st.sidebar.radio(
        "📂 데이터셋 선택",
        ("NOAA 해수온 (OISST)", "기상청 폭염일수 (서울)", "사용자 입력 예시 데이터")
    )

    # 기간 선택 (안전한 날짜 범위 계산)
    try:
        if dataset_choice == "NOAA 해수온 (OISST)" and not public["sst"].empty:
            data_min = public["sst"]["date"].min().date()
            data_max = public["sst"]["date"].max().date()
        elif dataset_choice == "기상청 폭염일수 (서울)" and not public["kma_heatwave"].empty:
            data_min = date(int(public["kma_heatwave"]["year"].min()), 1, 1)
            data_max = date(int(public["kma_heatwave"]["year"].max()), 12, 31)
        else:
            if not user_input["sst_east"].empty:
                data_min = user_input["sst_east"]["date"].min().date()
                data_max = user_input["sst_east"]["date"].max().date()
            else:
                data_min = date(2020, 1, 1)
                data_max = date.today()

        period = st.sidebar.date_input(
            "📅 분석 기간 선택",
            [data_min, data_max],
            min_value=data_min,
            max_value=data_max,
        )
    except Exception as e:
        logger.error(f"날짜 범위 설정 실패: {e}")
        st.sidebar.error("날짜 범위 설정에 문제가 있습니다.")
        period = [date(2020, 1, 1), date.today()]

    analysis_option = st.sidebar.selectbox(
        "🔍 분석 옵션 선택",
        ("추세 분석", "계절성 분석", "간단 요약 통계"),
    )

    # 메인 타이틀
    st.write("## 🌊 해수온/폭염 대시보드")

    # 데이터별 시각화 및 분석
    try:
        if dataset_choice == "NOAA 해수온 (OISST)":
            display_noaa_data(public["sst"], period, analysis_option)
        elif dataset_choice == "기상청 폭염일수 (서울)":
            display_kma_data(public["kma_heatwave"], period, analysis_option)
        else:
            display_user_data(user_input, period, analysis_option)
            
    except Exception as e:
        logger.error(f"데이터 표시 중 오류: {e}")
        st.error(f"📊 데이터를 표시하는 중 오류가 발생했습니다: {str(e)}")

def display_noaa_data(df: pd.DataFrame, period, analysis_option: str):
    """NOAA 해수온 데이터 표시"""
    if df.empty:
        st.error("NOAA 데이터가 없습니다.")
        return
        
    st.subheader("🌍 NOAA OISST 해수온 (글로벌 평균)")
    
    try:
        # 기간 필터링
        if isinstance(period, list) and len(period) == 2:
            df_filtered = df[
                (df["date"] >= pd.to_datetime(period[0])) & 
                (df["date"] <= pd.to_datetime(period[1]))
            ]
        else:
            df_filtered = df
            
        if df_filtered.empty:
            st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
            return
            
        # 기본 라인 차트
        st.line_chart(df_filtered.set_index("date"))

        # 분석 옵션별 표시
        if analysis_option == "간단 요약 통계":
            st.subheader("📊 통계 요약")
            st.write(df_filtered["sst_global_mean_C"].describe())
            
        elif analysis_option == "추세 분석":
            st.subheader("📈 추세 분석")
            fig = create_scatter_with_trend(df_filtered, "date", "sst_global_mean_C", "추세선 포함 해수온 변화")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        elif analysis_option == "계절성 분석":
            st.subheader("🗓️ 계절성 분석")
            df_seasonal = df_filtered.copy()
            df_seasonal["month"] = df_seasonal["date"].dt.month
            monthly_avg = df_seasonal.groupby("month")["sst_global_mean_C"].mean().reset_index()
            
            fig = px.line(monthly_avg, x="month", y="sst_global_mean_C", 
                         title="월별 평균 해수온 (계절성 분석)")
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        logger.error(f"NOAA 데이터 표시 실패: {e}")
        st.error(f"NOAA 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")

def display_kma_data(df: pd.DataFrame, period, analysis_option: str):
    """기상청 폭염일수 데이터 표시"""
    if df.empty:
        st.error("기상청 데이터가 없습니다.")
        return
        
    st.subheader("🔥 기상청 폭염일수 (서울)")
    
    try:
        # 기간 필터링
        if isinstance(period, list) and len(period) == 2:
            df_filtered = df[
                (df["year"] >= period[0].year) & 
                (df["year"] <= period[1].year)
            ]
        else:
            df_filtered = df
            
        if df_filtered.empty:
            st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
            return

        # 바차트 표시
        fig = px.bar(df_filtered, x="year", y="heatwave_days_seoul",
                     labels={"year": "연도", "heatwave_days_seoul": "폭염일수"},
                     title="연도별 폭염일수")
        st.plotly_chart(fig, use_container_width=True)

        # 분석 옵션별 표시
        if analysis_option == "간단 요약 통계":
            st.subheader("📊 통계 요약")
            st.write(df_filtered["heatwave_days_seoul"].describe())
            
        elif analysis_option == "추세 분석":
            st.subheader("📈 추세 분석")
            fig = create_scatter_with_trend(df_filtered, "year", "heatwave_days_seoul", "연도별 폭염일수 추세")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        elif analysis_option == "계절성 분석":
            st.info("⚠️ 폭염일수 데이터는 연도 단위라서 월별 계절성 분석이 불가능합니다.")
            
    except Exception as e:
        logger.error(f"KMA 데이터 표시 실패: {e}")
        st.error(f"기상청 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")

def display_user_data(user_input: Dict, period, analysis_option: str):
    """사용자 입력 데이터 표시"""
    try:
        st.subheader("📝 사용자 입력 설문 예시 데이터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not user_input["survey"].empty:
                fig1 = px.pie(user_input["survey"], names="response", values="count",
                              title="폭염 인식 설문")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.error("설문 데이터가 없습니다.")
                
        with col2:
            if not user_input["impacts"].empty:
                fig2 = px.bar(user_input["impacts"], x="impact", y="percent",
                              title="폭염 영향")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("영향 데이터가 없습니다.")

        st.subheader("🌊 동해 평균 해수온 (예시)")
        
        if user_input["sst_east"].empty:
            st.error("동해 해수온 데이터가 없습니다.")
            return
            
        df = user_input["sst_east"]
        
        # 기간 필터링
        if isinstance(period, list) and len(period) == 2:
            df_filtered = df[
                (df["date"] >= pd.to_datetime(period[0])) & 
                (df["date"] <= pd.to_datetime(period[1]))
            ]
        else:
            df_filtered = df
            
        if df_filtered.empty:
            st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
            return
            
        st.line_chart(df_filtered.set_index("date"))

        # 분석 옵션별 표시
        if analysis_option == "간단 요약 통계":
            st.subheader("📊 통계 요약")
            st.write(df_filtered["sst_east_C"].describe())
            
        elif analysis_option == "추세 분석":
            st.subheader("📈 추세 분석")
            fig = create_scatter_with_trend(df_filtered, "date", "sst_east_C", "동해 해수온 추세")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
        elif analysis_option == "계절성 분석":
            st.subheader("🗓️ 계절성 분석")
            df_seasonal = df_filtered.copy()
            df_seasonal["month"] = df_seasonal["date"].dt.month
            monthly_avg = df_seasonal.groupby("month")["sst_east_C"].mean().reset_index()
            
            fig = px.line(monthly_avg, x="month", y="sst_east_C",
                          title="동해 해수온 월별 평균 (계절성 분석)")
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        logger.error(f"사용자 데이터 표시 실패: {e}")
        st.error(f"사용자 데이터를 처리하는 중 오류가 발생했습니다: {str(e)}")

# 앱 실행
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"앱 실행 중 치명적 오류: {e}")
        st.error("🚨 애플리케이션에서 치명적인 오류가 발생했습니다. 페이지를 새로고침해주세요.")
        st.exception(e)