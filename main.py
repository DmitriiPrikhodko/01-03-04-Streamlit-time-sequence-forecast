import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn
from sklearn.metrics import *

import warnings

warnings.filterwarnings("ignore")

sklearn.set_config(transform_output="pandas")


def reset_form():
    st.session_state.prediction_made = False
    st.session_state.prediction_result = None
    st.session_state.show_form = True
    st.session_state.file_uploaded = False
    st.session_state.period = None
    st.session_state.show_period_selector = False
    st.session_state.resampled_data = None


# @st.cache_data
def upload_file(file):
    df = pd.read_csv(file, index_col=False)
    if len(df.columns.to_list()) == 2:
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        st.write(df.dtypes)
    else:
        st.markdown(
            """
    <div style="background-color: rgba(255, 0, 0, 0.3); padding: 8px; border-radius: 5px;">
        <h5 style="color: black; text-align: left;">Неправильная структура файла</h5>
    </div>
    """,
            unsafe_allow_html=True,
        )
        reset_form()
    return df


st.title(":blue[Предсказание будущего]")

if "prediction_made" not in st.session_state:
    st.session_state.prediction_made = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "show_form" not in st.session_state:
    st.session_state.show_form = True
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "test_df" not in st.session_state:
    st.session_state.test_df = None
if "period" not in st.session_state:
    st.session_state.period = "День"
if "show_period_selector" not in st.session_state:
    st.session_state.show_period_selector = False
if "resampled_data" not in st.session_state:
    st.session_state.resampled_data = None


if not st.session_state.prediction_made and st.session_state.show_form:
    st.write(
        "Загружаемый файл должен иметь 2 колонки:\n1) Timestamp - дата, время и т.д.\n2) Данные"
    )
    get_file = st.file_uploader("Загрузи CSV файл")

    if get_file is not None:
        st.session_state.test_df = upload_file(get_file)
        st.session_state.file_uploaded = True
    else:
        st.stop()


def choose_period():
    # Инициализация состояния в session_state
    if "period" not in st.session_state:
        st.session_state.period = "День"
    if "show_period_selector" not in st.session_state:
        st.session_state.show_period_selector = True

    # Словарь для преобразования периода в параметр resample
    period_translation = {
        "День": "D",
        "Неделя": "W",
        "Месяц": "M",
        "Год": "Y",
        "Секунда": "S",
        "Минута": "T",
        "Час": "H",
    }

    # Если нужно показать выбор периода
    if st.session_state.show_period_selector:
        selected_period = st.selectbox(
            "Группируем данные по периоду:",
            ["День", "Неделя", "Месяц", "Год", "Секунда", "Минута", "Час"],
            key="period_selector",
        )

        # Сохраняем выбранный период
        st.session_state.period = selected_period

        if st.button("Сохранить период", key="save_period"):
            # Выполняем resample и сохраняем результат
            data = st.session_state.test_df.copy()
            data.set_index("ds", inplace=True)
            resampled_data = data.resample(
                period_translation[st.session_state.period]
            ).sum()
            st.session_state.resampled_data = resampled_data
            st.session_state.show_period_selector = False
            st.rerun()

    else:
        # Показываем текущий период и кнопку для изменения
        st.write(f"**Текущий период:** {st.session_state.period}")

        if st.button("Изменить период", key="change_period"):
            st.session_state.show_period_selector = True
            st.rerun()

    return st.session_state.period


if st.session_state.file_uploaded:
    st.write("Выбери период группировки данных")
    period = choose_period()

    # Отображаем исходные данные
    st.write("#### Исходные данные")
    st.dataframe(st.session_state.test_df)

    # Отображаем обработанные данные, если они есть
    if st.session_state.resampled_data is not None:
        st.write(f"#### Данные после группировки по периоду: {period}")
        st.dataframe(st.session_state.resampled_data)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Исходных записей:** {len(st.session_state.test_df)}")
        with col2:
            st.write(f"**После группировки:** {len(st.session_state.resampled_data)}")
