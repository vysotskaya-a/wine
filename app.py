"""Streamlit demo for HW3: inference only; training lives in the notebook."""

import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


APP_DIR = Path(__file__).resolve().parent
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Алкоголь, %",
    "is_red": "Красное вино",
}
ACCENT = "#8c3152"


@st.cache_resource
def load_model(path: str, modified_ns: int):
    """One read-only fitted model is shared between server sessions."""
    return joblib.load(path)


@st.cache_data
def load_test_data(path: str, modified_ns: int) -> pd.DataFrame:
    """Data caching returns a separate copy to each caller."""
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, X: pd.DataFrame, model_modified_ns: int) -> np.ndarray:
    """The threshold is deliberately absent from the cache key."""
    return _model.predict_proba(X)[:, 1]


def prepare_batch(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Validate a CSV and return numeric features in the training order."""
    missing = [feature for feature in features if feature not in raw.columns]
    if missing:
        raise ValueError("Не хватает столбцов: " + ", ".join(missing))
    if raw.empty:
        raise ValueError("В файле нет строк с винами.")
    if "target" in raw.columns:
        raise ValueError("Удалите столбец target: для прогноза нужны только признаки вина.")
    try:
        X = raw.loc[:, features].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("Все признаки должны быть числами. Используйте точку как десятичный разделитель.") from exc
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("В признаках есть пропуски или бесконечные значения. Заполните их корректными числами.")
    if not X["is_red"].isin([0, 1]).all():
        raise ValueError("В столбце is_red допустимы только 0 (белое) и 1 (красное).")
    return X


def draw_metrics(y: pd.Series, probability: np.ndarray, threshold: float) -> None:
    predicted = (probability >= threshold).astype(int)
    precision = metrics.precision_score(y, predicted, zero_division=0)
    recall = metrics.recall_score(y, predicted, zero_division=0)
    f1 = metrics.f1_score(y, predicted, zero_division=0)
    columns = st.columns(4)
    for column, label, value in zip(
        columns,
        ["Precision", "Recall", "F1", "Отобрано вин"],
        [f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}", str(int(predicted.sum()))],
    ):
        column.metric(label, value)

    st.caption(
        "Precision — доля хороших среди рекомендованных вин. "
        "Recall — доля найденных хороших вин. F1 объединяет эти показатели."
    )
    matrix = metrics.confusion_matrix(y, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay(matrix, display_labels=["Остальные", "Хорошие"]).plot(
        ax=axes[0], cmap="RdPu", colorbar=False
    )
    axes[0].set(title="Матрица ошибок", xlabel="Прогноз", ylabel="Настоящий класс")

    fpr, tpr, _ = metrics.roc_curve(y, probability)
    axes[1].plot(fpr, tpr, color=ACCENT, label=f"ROC-AUC = {metrics.roc_auc_score(y, probability):.3f}")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", alpha=0.6)
    axes[1].scatter(fp / (fp + tn), recall, color="#207f79", s=70, zorder=3, label=f"Порог {threshold:.3f}")
    axes[1].set(title="ROC-кривая", xlabel="False positive rate", ylabel="True positive rate", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[1].legend(loc="lower right", fontsize=8)

    pr_precision, pr_recall, _ = metrics.precision_recall_curve(y, probability)
    axes[2].plot(pr_recall, pr_precision, color=ACCENT, label=f"AP = {metrics.average_precision_score(y, probability):.3f}")
    axes[2].axhline(float(y.mean()), linestyle="--", color="gray", alpha=0.6, label="Случайное ранжирование")
    # With no selected wines, precision is undefined. The PR convention uses (0, 1).
    marker_precision = precision if tp + fp else 1.0
    axes[2].scatter(recall, marker_precision, color="#207f79", s=70, zorder=3, label=f"Порог {threshold:.3f}")
    axes[2].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    axes[2].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if not predicted.any():
        st.info("При этом пороге вина не отобраны. Precision не определён: в метриках показан 0, на PR-кривой — условная конечная точка (0, 1).")
    st.write(
        f"**FP = {fp}:** посредственные вина попали в подборку. "
        f"**FN = {fn}:** хорошие вина пропущены."
    )
    st.info(
        "Если повысить порог, подборка станет меньше: модель потребует большей уверенности, "
        "но пропустит больше хороших вин. Если понизить порог, хороших вин будет найдено больше, "
        "однако в подборку может попасть больше посредственных. Precision на конкретной выборке "
        "не обязан меняться строго монотонно."
    )
    st.caption("Тестовые данные используются только для демонстрации оценки. Настройка порога в интерфейсе не переобучает модель и не заменяет выбор порога на валидации.")


def main() -> None:
    st.set_page_config(page_title="Винная подборка · HW3", page_icon="🍷", layout="wide")
    st.title("🍷 Винная подборка")
    st.write("Оцените вино по лабораторным показателям и найдите баланс между качеством и широтой подборки.")
    st.caption("Хорошее вино: экспертная оценка не ниже 7. Прогноз построен логистической регрессией.")

    required = [APP_DIR / filename for filename in ("model.joblib", "config.json", "test.csv")]
    if any(not path.is_file() for path in required):
        st.error("Не найдены model.joblib, config.json или test.csv. Выполните ячейку сохранения модели в HW3_MMDA.ipynb и поместите файлы рядом с app.py.")
        st.stop()
    model_path, config_path, test_path = required
    model_version = model_path.stat().st_mtime_ns
    model = load_model(str(model_path), model_version)
    data = load_test_data(str(test_path), test_path.stat().st_mtime_ns)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    features = list(model.feature_names_in_)
    X_test = data.loc[:, features]
    y_test = data["target"]
    probability = test_proba(model, X_test, model_version)

    with st.sidebar:
        st.header("Правило отбора")
        threshold = st.slider(
            "Порог вероятности", min_value=0.0, max_value=1.0,
            value=float(config["threshold"]), step=0.001, format="%.3f", key="threshold",
            help="Вино попадает в подборку, если вероятность не ниже порога. Один порог действует на всех вкладках.",
        )
        st.caption(f"Порог из ноутбука: {float(config['threshold']):.4f}")
        st.write(f"**{len(data):,}** вин в тестовой выборке".replace(",", " "))
        st.write(f"**{y_test.mean():.1%}** из них — хорошие")

    single, evaluation, batch, coefficients = st.tabs([
        "Оценка одного вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"
    ])
    with single:
        st.subheader("Лабораторные показатели")
        st.caption("Начальные значения — медианы тестовой выборки. Диапазоны ввода соответствуют наблюдавшимся значениям.")
        with st.form("wine_features"):
            values = {}
            columns = st.columns(3)
            numeric_features = [name for name in features if name != "is_red"]
            for i, feature in enumerate(numeric_features):
                series = X_test[feature]
                with columns[i % 3]:
                    step = 0.00001 if feature == "density" else (0.001 if feature == "chlorides" else 0.01)
                    values[feature] = st.number_input(
                        LABELS.get(feature, feature), min_value=float(series.min()), max_value=float(series.max()),
                        value=float(series.median()), step=step,
                        format="%.5f" if feature == "density" else "%.3f",
                        help=feature, key=f"input_{feature}",
                    )
            values["is_red"] = int(st.toggle("Красное вино", value=bool(X_test["is_red"].median())))
            submitted = st.form_submit_button("Оценить вино", type="primary")
        if submitted or "single_wine" not in st.session_state:
            st.session_state["single_wine"] = values
        wine = pd.DataFrame([st.session_state["single_wine"]], columns=features)
        score = float(model.predict_proba(wine)[0, 1])
        st.metric("Вероятность хорошего вина", f"{score:.1%}")
        if score >= threshold:
            st.success(f"Добавить в подборку · вероятность {score:.3f} ≥ порог {threshold:.3f}")
        else:
            st.warning(f"Пропустить · вероятность {score:.3f} < порог {threshold:.3f}")
        st.caption("После изменения лабораторных показателей нажмите «Оценить вино». Изменение порога сразу обновляет решение.")

    with evaluation:
        st.subheader("Модельные метрики")
        draw_metrics(y_test, probability, threshold)

    with batch:
        st.subheader("Проверьте свою подборку")
        st.write("Загрузите CSV с 12 признаками, без target. Разделитель — запятая; is_red: 0 для белого и 1 для красного.")
        st.download_button("Скачать пример CSV", X_test.head(5).to_csv(index=False).encode("utf-8"), "wine_example.csv", "text/csv")
        uploaded = st.file_uploader("CSV с признаками вин", type=["csv"])
        if uploaded is not None:
            try:
                raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
                X_batch = prepare_batch(raw, features)
                scores = model.predict_proba(X_batch)[:, 1]
                result = raw.copy()
                result["probability_good"] = scores
                result["prediction"] = (scores >= threshold).astype(int)
                result["decision"] = np.where(result["prediction"] == 1, "Добавить в подборку", "Пропустить")
            except (ValueError, TypeError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
                st.error(f"Не удалось обработать файл. {exc}")
            else:
                st.caption(f"Обработано {len(result)} вин. Текущий порог: {threshold:.3f}.")
                st.dataframe(result, width="stretch", hide_index=True)
                st.download_button("Скачать прогнозы", result.to_csv(index=False).encode("utf-8-sig"), "wine_predictions.csv", "text/csv")
        with st.expander("Названия обязательных столбцов"):
            st.code(",".join(features), language=None)

    with coefficients:
        st.subheader("Какие признаки влияют на оценку")
        coef = pd.Series(model.steps[-1][1].coef_[0], index=features).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh([LABELS.get(name, name) for name in coef.index], coef.values, color=np.where(coef.values >= 0, "#207f79", ACCENT))
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Коэффициент при стандартизованном признаке")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        positive = coef[coef > 0].nlargest(3)
        negative = coef[coef < 0].nsmallest(3)
        st.write("**Сильнее всего за хорошее вино:** " + ", ".join(LABELS.get(name, name) for name in positive.index) + ".")
        st.write("**Сильнее всего против:** " + ", ".join(LABELS.get(name, name) for name in negative.index) + ".")
        st.caption("Коэффициент показывает изменение логарифма шансов при росте признака на одно стандартное отклонение и фиксированных остальных признаках. Это связь в данных, а не доказательство причинного влияния.")

    st.divider()
    st.caption("HW3 · Wine Quality · P. Cortez et al., 2009 · CC BY 4.0")


if __name__ == "__main__":
    main()
