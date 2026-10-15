
from pathlib import Path
import uuid

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.metrics import structural_similarity


# =========================================================
# НАСТРОЙКИ
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = BASE_DIR / "catalog" / "images"
CSV_PATH = DATA_DIR / "coins.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "id", "name", "region", "ruler", "date",
    "mint", "material", "weight", "reference", "image"
]

st.set_page_config(
    page_title="CoinAtlas — каталог монет",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# СТИЛИ
# =========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    .stApp {
        background: #f7f4ee;
        color: #292722;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #332c28;
    }

    [data-testid="stSidebar"] {
        background: #292725;
    }

    [data-testid="stSidebar"] * {
        color: #f7f1e8;
    }

    [data-testid="stSidebar"] label {
        color: #f7f1e8 !important;
    }

    .brand {
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 32px;
        font-weight: 700;
        color: #f2dfc0;
        padding: 12px 0 0 0;
    }

    .brand-subtitle {
        color: #c9bca9;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 28px;
    }

    .eyebrow {
        color: #8c6555;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero {
        padding: 24px 0 14px 0;
    }

    .hero p {
        color: #777168;
        font-size: 16px;
        max-width: 720px;
    }

    .metric-card {
        background: #fffdf8;
        border: 1px solid #e9e1d5;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 100px;
    }

    .metric-label {
        color: #81776c;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #3b302b;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 30px;
        margin-top: 4px;
    }

    .coin-card {
        background: #fffdf8;
        border: 1px solid #e8dfd3;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 8px;
    }

    .coin-name {
        color: #352c27;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .muted {
        color: #81786d;
        font-size: 13px;
    }

    .detail-panel {
        background: #fffdf8;
        border: 1px solid #e8dfd3;
        border-radius: 16px;
        padding: 24px;
    }

    .stButton > button {
        border-radius: 9px;
        min-height: 42px;
        font-weight: 600;
        border: 1px solid #75483f;
    }

    .stButton > button[kind="primary"] {
        background: #75483f;
        color: #ffffff;
        border: 1px solid #75483f;
    }

    .stButton > button[kind="primary"]:hover {
        background: #5d382f;
        border-color: #5d382f;
        color: #ffffff;
    }

    div[data-testid="stForm"] {
        background: #fffdf8;
        padding: 20px;
        border: 1px solid #e8dfd3;
        border-radius: 14px;
    }

    hr {
        border-color: #e4dbce;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# РАБОТА С ДАННЫМИ
# =========================================================

def load_coins():
    if not CSV_PATH.exists():
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(CSV_PATH, index=False)
        return df

    try:
        df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=COLUMNS)

    for column in COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df[COLUMNS]


def save_coins(df):
    df = df[COLUMNS].fillna("")
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def get_image_path(filename):
    if not filename:
        return None

    path = (IMAGE_DIR / str(filename)).resolve()

    if path.parent != IMAGE_DIR.resolve():
        return None

    if path.exists() and path.is_file():
        return path

    return None


def save_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return ""

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        suffix = ".jpg"

    filename = f"{uuid.uuid4().hex}{suffix}"
    destination = IMAGE_DIR / filename

    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    decoded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if decoded is None:
        raise ValueError("Не удалось прочитать изображение.")

    if not cv2.imwrite(str(destination), decoded):
        raise ValueError("Не удалось сохранить изображение.")

    return filename


def remove_unused_image(filename, df):
    if not filename:
        return

    still_used = (
        not df.empty
        and (df["image"] == filename).any()
    )

    if not still_used:
        path = get_image_path(filename)
        if path:
            try:
                path.unlink()
            except OSError:
                pass


def find_coin(df, coin_id):
    matches = df[df["id"].astype(str) == str(coin_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def unique_coin_id(df, proposed_id):
    proposed_id = str(proposed_id).strip()

    if not proposed_id:
        proposed_id = "coin"

    existing = set(df["id"].astype(str).str.lower())
    candidate = proposed_id
    counter = 2

    while candidate.lower() in existing:
        candidate = f"{proposed_id}-{counter}"
        counter += 1

    return candidate


# =========================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЙ
# Оставлены только два метода:
# 1. Адаптивная бинаризация
# 2. Обнаружение границ
# =========================================================

def read_grayscale_image(image_path):
    if image_path is None:
        return None

    image = cv2.imread(str(image_path))
    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_binarization(gray_image):
    # Лёгкое сглаживание снижает влияние мелкого шума.
    blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Адаптивная бинаризация рассчитывает порог
    # отдельно для локальных участков изображения.
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    return binary


def detect_edges(gray_image):
    # Автоматически подбираем пороги Canny
    # на основе медианной яркости изображения.
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    median = float(np.median(blurred))

    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))

    if upper <= lower:
        lower, upper = 50, 150

    edges = cv2.Canny(blurred, lower, upper)
    return edges


# =========================================================
# СРАВНЕНИЕ МОНЕТ
# =========================================================

def prepare_comparison_image(image_bytes, size=(300, 300)):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(gray)

    return gray


def compare_coin_images(image_a, image_b):
    if image_a is None or image_b is None:
        return 0.0

    try:
        ssim_score = structural_similarity(
            image_a,
            image_b,
            data_range=255,
        )
    except Exception:
        ssim_score = 0.0

    hist_a = cv2.calcHist([image_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_b], [0], None, [256], [0, 256])

    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    correlation = cv2.compareHist(
        hist_a,
        hist_b,
        cv2.HISTCMP_CORREL,
    )

    # Преобразуем корреляцию в диапазон 0–1.
    correlation = max(0.0, min(1.0, (correlation + 1.0) / 2.0))
    ssim_score = max(0.0, min(1.0, float(ssim_score)))

    return 100 * (0.7 * ssim_score + 0.3 * correlation)


# =========================================================
# СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# =========================================================

if "page" not in st.session_state:
    st.session_state.page = "Коллекция"

if "selected_coin_id" not in st.session_state:
    st.session_state.selected_coin_id = ""

if "comparison_upload" not in st.session_state:
    st.session_state.comparison_upload = None


def go_to(page):
    st.session_state.page = page
    st.rerun()


coins = load_coins()


# =========================================================
# БОКОВАЯ ПАНЕЛЬ
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">CoinAtlas</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-subtitle">Нумизматический каталог</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**НАВИГАЦИЯ**")

    navigation = {
        "Коллекция": "Коллекция",
        "О проекте": "О проекте",
    }

    for label, page_name in navigation.items():
        is_active = st.session_state.page == page_name

        if st.button(
            label,
            key=f"nav_{page_name}",
            use_container_width=True,
            type="primary",
        ):
            go_to(page_name)

    st.divider()

    st.markdown("**КАТАЛОГ**")
    st.markdown(f"Монет в коллекции: **{len(coins)}**")

    if st.button(
        "＋ Добавить монету",
        key="sidebar_add_coin",
        use_container_width=True,
        type="primary",
    ):
        go_to("Добавить монету")

    if st.button(
        "⚙ Управление каталогом",
        key="sidebar_manage",
        use_container_width=True,
        type="primary",
    ):
        go_to("Управление каталогом")

# =========================================================
# СТРАНИЦА: КОЛЛЕКЦИЯ
# =========================================================

if st.session_state.page == "Коллекция":
    st.markdown('<div class="eyebrow">Нумизматическое собрание</div>', unsafe_allow_html=True)

    header_left, header_right = st.columns([3, 1])

    with header_left:
        st.title("Коллекция монет")
        st.markdown(
            '<div class="muted">Изучайте предметы, открывайте подробные карточки '
            'и сравнивайте изображения.</div>',
            unsafe_allow_html=True,
        )

    with header_right:
        st.write("")
        st.write("")
        if st.button(
            "⇄ Сравнить монеты",
            type="primary",
            use_container_width=True,
            key="main_compare_button",
        ):
            go_to("Сравнение монет")

    st.divider()

    regions_count = (
        coins["region"].replace("", pd.NA).nunique()
        if not coins.empty else 0
    )

    materials_count = (
        coins["material"].replace("", pd.NA).nunique()
        if not coins.empty else 0
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Всего монет</div>'
            f'<div class="metric-value">{len(coins)}</div></div>',
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Регионов</div>'
            f'<div class="metric-value">{regions_count}</div></div>',
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Материалов</div>'
            f'<div class="metric-value">{materials_count}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")

    filter_left, filter_right = st.columns([2, 1])

    with filter_left:
        search = st.text_input(
            "Поиск по каталогу",
            placeholder="Название, правитель, монетный двор, дата…",
            key="catalog_search",
        )

    with filter_right:
        region_options = ["Все регионы"] + sorted(
            [r for r in coins["region"].unique().tolist() if r]
        ) if not coins.empty else ["Все регионы"]

        selected_region = st.selectbox(
            "Регион",
            region_options,
            key="catalog_region",
        )

    filtered = coins.copy()

    if search:
        search_lower = search.lower()
        searchable_columns = [
            "id", "name", "region", "ruler", "date",
            "mint", "material", "reference"
        ]

        mask = filtered[searchable_columns].apply(
            lambda column: column.astype(str).str.lower().str.contains(
                search_lower, regex=False
            )
        ).any(axis=1)

        filtered = filtered[mask]

    if selected_region != "Все регионы":
        filtered = filtered[filtered["region"] == selected_region]

    st.write("")

    if filtered.empty:
        st.info("Монеты не найдены. Попробуйте изменить параметры поиска.")
    else:
        st.caption(f"Найдено монет: {len(filtered)}")

        for start in range(0, len(filtered), 3):
            row = filtered.iloc[start:start + 3]
            card_columns = st.columns(3)

            for column, (_, coin) in zip(card_columns, row.iterrows()):
                with column:
                    image_path = get_image_path(coin["image"])

                    st.markdown('<div class="coin-card">', unsafe_allow_html=True)

                    if image_path:
                        st.image(str(image_path), use_container_width=True)
                    else:
                        st.markdown(
                            '<div style="height:160px;display:flex;'
                            'align-items:center;justify-content:center;'
                            'background:#eee7dc;border-radius:10px;'
                            'color:#8b7d6d;">Изображение отсутствует</div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown(
                        f'<div class="coin-name">{coin["name"] or "Без названия"}</div>',
                        unsafe_allow_html=True,
                    )

                    details = " · ".join(
                        value for value in [coin["region"], coin["date"]]
                        if value
                    )
                    st.markdown(
                        f'<div class="muted">{details or "Нет описания"}</div>',
                        unsafe_allow_html=True,
                    )

                    st.write("")

                    if st.button(
                        "Открыть карточку →",
                        key=f"open_coin_{coin['id']}",
                        use_container_width=True,
                        type="primary",
                    ):
                        st.session_state.selected_coin_id = str(coin["id"])
                        go_to("Карточка монеты")

                    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# СТРАНИЦА: КАРТОЧКА МОНЕТЫ
# =========================================================

elif st.session_state.page == "Карточка монеты":
    if st.button("← Вернуться в коллекцию", key="back_to_collection"):
        go_to("Коллекция")

    coin = find_coin(coins, st.session_state.selected_coin_id)

    if coin is None:
        st.warning("Монета не найдена в каталоге.")
        if st.button("Открыть коллекцию"):
            go_to("Коллекция")
    else:
        st.markdown('<div class="eyebrow">Карточка предмета</div>', unsafe_allow_html=True)
        st.title(coin["name"] or "Монета без названия")

        image_path = get_image_path(coin["image"])
        left, right = st.columns([1.1, 1])

        with left:
            if image_path:
                st.image(str(image_path), use_container_width=True)
            else:
                st.info("Для этой монеты изображение не добавлено.")

        with right:
            st.markdown('<div class="detail-panel">', unsafe_allow_html=True)
            st.subheader("Описание предмета")

            fields = [
                ("Идентификатор", "id"),
                ("Регион", "region"),
                ("Правитель", "ruler"),
                ("Дата / период", "date"),
                ("Монетный двор", "mint"),
                ("Материал", "material"),
                ("Вес", "weight"),
                ("Справочник / источник", "reference"),
            ]

            for label, field in fields:
                value = str(coin.get(field, "")).strip()
                if value:
                    st.markdown(f"**{label}:** {value}")

            st.markdown("</div>", unsafe_allow_html=True)

        st.divider()
        st.subheader("Обработка изображения")
        st.markdown(
            "Выберите один из двух методов компьютерного зрения. "
            "Результат можно использовать для изучения рельефа и контуров "
            "на фотографии монеты."
        )

        if not image_path:
            st.info("Добавьте изображение монеты, чтобы использовать обработку.")
        else:
            gray = read_grayscale_image(image_path)

            if gray is None:
                st.error("Не удалось прочитать изображение монеты.")
            else:
                method = st.radio(
                    "Метод обработки",
                    [
                        "3. Адаптивная бинаризация",
                        "4. Обнаружение границ",
                    ],
                    horizontal=True,
                    key="coin_image_method",
                )

                if method == "3. Адаптивная бинаризация":
                    result = adaptive_binarization(gray)
                    st.markdown("#### Адаптивная бинаризация")
                    st.markdown(
                        "Преобразует изображение в чёрно-белое, подбирая порог "
                        "для локальных областей. Может помочь выделить элементы "
                        "изображения при неоднородном освещении."
                    )
                else:
                    result = detect_edges(gray)
                    st.markdown("#### Обнаружение границ")
                    st.markdown(
                        "Алгоритм Canny выделяет границы объектов и переходы "
                        "яркости. Это может помочь изучить контуры изображения "
                        "и детали рельефа."
                    )

                original_col, processed_col = st.columns(2)

                with original_col:
                    st.caption("Исходное изображение")
                    st.image(gray, use_container_width=True)

                with processed_col:
                    st.caption("Результат обработки")
                    st.image(result, use_container_width=True)

                st.download_button(
                    "Скачать результат обработки",
                    data=cv2.imencode(".png", result)[1].tobytes(),
                    file_name=f"{coin['id']}_{'binary' if method.startswith('3.') else 'edges'}.png",
                    mime="image/png",
                )

        st.write("")
        if st.button("✎ Редактировать эту монету", type="primary"):
            st.session_state.edit_coin_id = str(coin["id"])
            go_to("Управление каталогом")


# =========================================================
# СТРАНИЦА: ДОБАВИТЬ МОНЕТУ
# =========================================================

elif st.session_state.page == "Добавить монету":
    st.markdown('<div class="eyebrow">Новая запись</div>', unsafe_allow_html=True)
    st.title("Добавить монету")
    st.markdown("Заполните сведения о предмете и при необходимости загрузите фотографию.")

    with st.form("add_coin_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Название монеты *")
            region = st.text_input("Регион")
            ruler = st.text_input("Правитель")
            date = st.text_input("Дата / период")

        with col2:
            mint = st.text_input("Монетный двор")
            material = st.text_input("Материал")
            weight = st.text_input("Вес")
            reference = st.text_input("Справочник / источник")

        uploaded_image = st.file_uploader(
            "Фотография монеты",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
        )

        submitted = st.form_submit_button(
            "Сохранить монету",
            type="primary",
            use_container_width=True,
        )

        if submitted:
            if not name.strip():
                st.error("Укажите название монеты.")
            else:
                new_id = unique_coin_id(coins, name.strip().lower().replace(" ", "-"))

                try:
                    image_filename = save_uploaded_image(uploaded_image)

                    new_coin = {
                        "id": new_id,
                        "name": name.strip(),
                        "region": region.strip(),
                        "ruler": ruler.strip(),
                        "date": date.strip(),
                        "mint": mint.strip(),
                        "material": material.strip(),
                        "weight": weight.strip(),
                        "reference": reference.strip(),
                        "image": image_filename,
                    }

                    updated = pd.concat(
                        [coins, pd.DataFrame([new_coin])],
                        ignore_index=True,
                    )
                    save_coins(updated)

                    st.session_state.selected_coin_id = new_id
                    st.session_state.page = "Карточка монеты"
                    st.success("Монета добавлена в каталог.")
                    st.rerun()

                except Exception as error:
                    st.error(f"Не удалось сохранить монету: {error}")


# =========================================================
# СТРАНИЦА: СРАВНЕНИЕ МОНЕТ
# =========================================================

elif st.session_state.page == "Сравнение монет":
    st.markdown('<div class="eyebrow">Компьютерное зрение</div>', unsafe_allow_html=True)
    st.title("Сравнение монет")
    st.markdown(
        "Загрузите изображение. Система сопоставит его с фотографиями "
        "из каталога и покажет наиболее визуально похожие варианты."
    )

    if st.button("← Вернуться в коллекцию"):
        go_to("Коллекция")

    uploaded = st.file_uploader(
        "Изображение для сравнения",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="compare_upload",
    )

    if uploaded is not None:
        st.image(uploaded, caption="Изображение для поиска", width=300)

        if st.button("Найти похожие монеты", type="primary"):
            query_image = prepare_comparison_image(uploaded.getvalue())

            if query_image is None:
                st.error("Не удалось прочитать загруженное изображение.")
            elif coins.empty:
                st.info("Каталог пока пуст.")
            else:
                results = []

                for _, coin in coins.iterrows():
                    image_path = get_image_path(coin["image"])

                    if not image_path:
                        continue

                    try:
                        candidate_bytes = image_path.read_bytes()
                        candidate_image = prepare_comparison_image(candidate_bytes)

                        if candidate_image is None:
                            continue

                        score = compare_coin_images(
                            query_image,
                            candidate_image,
                        )

                        results.append({
                            "coin": coin,
                            "score": score,
                            "image_path": image_path,
                        })
                    except Exception:
                        continue

                results.sort(key=lambda item: item["score"], reverse=True)
                results = results[:3]

                if not results:
                    st.warning("В каталоге нет доступных изображений для сравнения.")
                else:
                    st.subheader("Наиболее похожие изображения")
                    st.caption(
                        "Процент показывает визуальное сходство изображений, "
                        "а не вероятность того, что это одна и та же монета."
                    )

                    result_columns = st.columns(len(results))

                    for column, result in zip(result_columns, results):
                        coin = result["coin"]

                        with column:
                            st.image(
                                str(result["image_path"]),
                                use_container_width=True,
                            )
                            st.markdown(f"**{coin['name']}**")
                            st.caption(
                                " · ".join(
                                    value for value in [
                                        coin["region"], coin["date"]
                                    ] if value
                                )
                            )
                            st.progress(
                                min(1.0, max(0.0, result["score"] / 100))
                            )
                            st.write(f"Визуальное сходство: **{result['score']:.1f}%**")

                            if st.button(
                                "Открыть карточку",
                                key=f"comparison_open_{coin['id']}",
                                use_container_width=True,
                            ):
                                st.session_state.selected_coin_id = str(coin["id"])
                                go_to("Карточка монеты")


# =========================================================
# СТРАНИЦА: УПРАВЛЕНИЕ КАТАЛОГОМ
# =========================================================

elif st.session_state.page == "Управление каталогом":
    st.markdown('<div class="eyebrow">Администрирование</div>', unsafe_allow_html=True)
    st.title("Управление каталогом")

    if coins.empty:
        st.info("Каталог пуст. Добавьте первую монету.")
    else:
        coin_options = coins["id"].astype(str).tolist()

        default_id = st.session_state.get("edit_coin_id", "")
        if default_id not in coin_options:
            default_id = coin_options[0]

        selected_id = st.selectbox(
            "Выберите монету для редактирования",
            coin_options,
            index=coin_options.index(default_id),
            format_func=lambda coin_id: (
                f"{coin_id} — "
                f"{find_coin(coins, coin_id)['name'] or 'Без названия'}"
            ),
        )

        selected_coin = find_coin(coins, selected_id)

        st.subheader("Редактирование")

        with st.form("edit_coin_form"):
            col1, col2 = st.columns(2)

            with col1:
                edit_name = st.text_input("Название", value=selected_coin["name"])
                edit_region = st.text_input("Регион", value=selected_coin["region"])
                edit_ruler = st.text_input("Правитель", value=selected_coin["ruler"])
                edit_date = st.text_input("Дата / период", value=selected_coin["date"])

            with col2:
                edit_mint = st.text_input("Монетный двор", value=selected_coin["mint"])
                edit_material = st.text_input("Материал", value=selected_coin["material"])
                edit_weight = st.text_input("Вес", value=selected_coin["weight"])
                edit_reference = st.text_input(
                    "Справочник / источник",
                    value=selected_coin["reference"],
                )

            replacement_image = st.file_uploader(
                "Заменить фотографию (необязательно)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )

            save_edit = st.form_submit_button(
                "Сохранить изменения",
                type="primary",
                use_container_width=True,
            )

            if save_edit:
                if not edit_name.strip():
                    st.error("Название монеты не может быть пустым.")
                else:
                    updated = coins.copy()
                    index = updated.index[
                        updated["id"].astype(str) == str(selected_id)
                    ][0]

                    old_image = updated.at[index, "image"]
                    new_image = old_image

                    try:
                        if replacement_image is not None:
                            new_image = save_uploaded_image(replacement_image)

                        updated.at[index, "name"] = edit_name.strip()
                        updated.at[index, "region"] = edit_region.strip()
                        updated.at[index, "ruler"] = edit_ruler.strip()
                        updated.at[index, "date"] = edit_date.strip()
                        updated.at[index, "mint"] = edit_mint.strip()
                        updated.at[index, "material"] = edit_material.strip()
                        updated.at[index, "weight"] = edit_weight.strip()
                        updated.at[index, "reference"] = edit_reference.strip()
                        updated.at[index, "image"] = new_image

                        save_coins(updated)

                        if new_image != old_image:
                            remove_unused_image(old_image, updated)

                        st.success("Изменения сохранены.")
                        st.rerun()

                    except Exception as error:
                        st.error(f"Не удалось сохранить изменения: {error}")

        st.divider()
        st.subheader("Удаление монеты")

        st.warning(
            f"Будет удалена запись «{selected_coin['name']}» "
            "из каталога. Это действие нельзя отменить."
        )

        confirm_delete = st.checkbox(
            "Подтверждаю удаление этой монеты",
            key=f"confirm_delete_{selected_id}",
        )

        if st.button(
            "Удалить монету",
            type="primary",
            disabled=not confirm_delete,
        ):
            old_image = selected_coin["image"]
            updated = coins[coins["id"].astype(str) != str(selected_id)].copy()
            save_coins(updated)
            remove_unused_image(old_image, updated)

            st.session_state.selected_coin_id = ""
            st.session_state.page = "Управление каталогом"
            st.success("Монета удалена.")
            st.rerun()


# =========================================================
# СТРАНИЦА: О ПРОЕКТЕ
# =========================================================

elif st.session_state.page == "О проекте":
    st.markdown('<div class="eyebrow">О проекте</div>', unsafe_allow_html=True)
    st.title("CoinAtlas")
    st.markdown(
        """
        **CoinAtlas** — цифровой каталог для систематизации и визуального
        изучения нумизматических объектов.

        ### Возможности

        - Каталог монет с фотографиями и описаниями.
        - Поиск по названию, региону, правителю, дате и другим полям.
        - Добавление, редактирование и удаление записей.
        - Сравнение загруженного изображения с изображениями каталога.
        - Адаптивная бинаризация для выделения деталей изображения.
        - Обнаружение границ методом Canny.

        ### Используемые технологии

        - **Python** — логика приложения.
        - **Streamlit** — пользовательский интерфейс.
        - **OpenCV** — обработка изображений.
        - **NumPy и Pandas** — работа с данными.
        - **scikit-image** — вычисление структурного сходства изображений.

        Результаты обработки и визуального сравнения предназначены для
        исследовательской работы и не заменяют экспертную атрибуцию монеты.
        """
    )
