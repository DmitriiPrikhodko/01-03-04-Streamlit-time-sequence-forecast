import numpy as np
import pandas as pd
import streamlit as st
import sklearn
from sklearn.metrics import *
import plotly.express as px
import warnings
from prophet import Prophet
import itertools
import logging
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 300

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
    st.session_state.forecast_period = None
    st.session_state.show_period_input = False


def params_tuning(data_train, data_test, number_of_future_predicted_points):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)  # Выключение логов

    param_grid = {
        "changepoint_prior_scale": list(np.arange(0.01, 0.5, 0.04)),
        "seasonality_prior_scale": list(np.arange(1, 10, 0.5)),
    }

    # Generate all combinations of parameters
    all_params = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]
    maes = []  # Store the MAEs for each params here

    # Use validation to evaluate all parameters
    for params in all_params:
        model = Prophet(**params).fit(
            data_train
        )  # Обучаем модель с заданными параметрами
        future = model.make_future_dataframe(
            periods=number_of_future_predicted_points, freq="M"
        )  # Создаем фрейм данных для будущего (для тестовой выборки)
        forecast = model.predict(future)  # Делаем предсказание на будущее
        forecast_test = forecast[
            -number_of_future_predicted_points : -number_of_future_predicted_points
            + len(data_test)
        ]
        prophet_mae_test = np.round(
            mean_absolute_error(data_prophet_test["y"], forecast_test["yhat"]), 1
        )
        maes.append(prophet_mae_test)  # Добавляем RMSE для данного набора параметров

    # Find the best parameters
    tuning_results = pd.DataFrame(all_params)
    tuning_results["maes"] = maes
    par1 = tuning_results.sort_values(by="maes").iloc[0, 0]
    par2 = tuning_results.sort_values(by="maes").iloc[0, 0]
    return par1, par2


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
st.write(
    "тестовый датасет https://github.com/DmitriiPrikhodko/01-03-04-Streamlit-time-sequence-forecast/blob/main/test_data/full_set.csv"
)

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
if "forecast_period" not in st.session_state:
    st.session_state.forecast_period = None
if "show_period_input" not in st.session_state:
    st.session_state.show_period_input = False


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


def calculate_metrics(y_true, y_pred):
    """Расчет различных метрик качества"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def choose_period():
    # Инициализация состояния в session_state
    if "period" not in st.session_state:
        st.session_state.period = "День"
    if "show_period_selector" not in st.session_state:
        st.session_state.show_period_selector = True

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

        if st.button("Построить интерактивный график"):
            try:
                if isinstance(st.session_state.resampled_data, pd.Series):
                    # Для Series
                    fig = px.line(
                        st.session_state.resampled_data,
                        title="Ресемплированные данные",
                    )
                    fig.update_layout(
                        xaxis_title="Время",
                        yaxis_title="Значение",
                        hovermode="x unified",
                    )

                elif isinstance(st.session_state.resampled_data, pd.DataFrame):
                    # Для DataFrame
                    fig = px.line(
                        st.session_state.resampled_data,
                        title="Ресемплированные данные",
                    )
                    fig.update_layout(
                        xaxis_title="Время",
                        yaxis_title="Значение",
                        hovermode="x unified",
                    )

                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при построении графика: {e}")
        if st.session_state.show_period_input:
            st.write("Введите количество периодов предсказания")
            st.write(f"Текущий период - {st.session_state.get('period', 'не указан')}")

            n = st.number_input(
                "Кол-во периодов",
                min_value=0,
                max_value=500,
                step=1,
                key="period_input",
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Сохранить", key="save_period"):
                    st.session_state.forecast_period = n
                    st.session_state.show_period_input = False
                    st.rerun()

            with col2:
                if st.button("Отмена", key="cancel_period"):
                    st.session_state.show_period_input = False
                    st.rerun()
        # Если период сохранен, показываем возможность изменить
        elif st.session_state.forecast_period is not None:
            st.write(f"Текущее количество периодов: {st.session_state.forecast_period}")

            if st.button("Изменить количество периодов", key="change_period_count"):
                st.session_state.forecast_period = None
                st.session_state.show_period_input = True
                st.rerun()

        if st.button("Сделать прогноз", key="make_forecast"):
            if st.session_state.forecast_period is None:
                st.session_state.show_period_input = True
            else:
                # Выполнить прогноз с сохраненным периодом
                st.write(
                    f"Выполняем прогноз на {st.session_state.forecast_period} периодов..."
                )
                data_prophet_train = st.session_state.resampled_data.iloc[
                    : int(len(st.session_state.resampled_data.index) * 0.8), :
                ]
                data_prophet_train = data_prophet_train.reset_index()
                data_prophet_train.columns = ["ds", "y"]

                data_prophet_test = st.session_state.resampled_data.iloc[
                    int(len(st.session_state.resampled_data.index) * 0.8 + 1) :, :
                ]
                data_prophet_test = data_prophet_test.reset_index()
                data_prophet_test.columns = ["ds", "y"]
                # st.write(*data_prophet_test.columns)
                # st.dataframe(data_prophet_test)
                model = Prophet(
                    changepoint_prior_scale=params_tuning(
                        data_prophet_train,
                        data_prophet_test,
                        st.session_state.forecast_period,
                    )[0],
                    seasonality_prior_scale=params_tuning(
                        data_prophet_train,
                        data_prophet_test,
                        st.session_state.forecast_period,
                    )[1],
                )
                model.fit(data_prophet_train)
                future = model.make_future_dataframe(
                    periods=st.session_state.forecast_period,
                    freq=period_translation[st.session_state.period],
                )
                forecast = model.predict(future)
                fig_prophet = model.plot(forecast)
                fig_prophet.legend(
                    fontsize=14,
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                )
                plt.title("Прогноз данных с помощью Prophet", fontsize=16)
                plt.xlabel("Дата", fontsize=12)
                plt.ylabel("Значение", fontsize=12)
                plt.tight_layout()
                ax = fig_prophet.gca()  # Получаем текущие оси
                ax.plot(
                    st.session_state.resampled_data.index,
                    st.session_state.resampled_data.values,
                    "k-",
                    linewidth=0.5,
                    label="Observed data",
                )
                st.pyplot(fig_prophet)
                # forecast_train = model.predict(data_prophet_train["ds"])
                forecast_test = model.predict(data_prophet_test[["ds"]])
                st.write("Метрики модели")
                st.dataframe(
                    pd.DataFrame(
                        data=calculate_metrics(
                            forecast_test["yhat"], data_prophet_test["y"]
                        ),
                        index=["ProphetModel"],
                    )
                )
                st.write("Компоненты модели (тренд и годовой срез)")
                figure_comp = model.plot_components(forecast)
                st.pyplot(figure_comp)
