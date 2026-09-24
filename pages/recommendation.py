"""
Mô-đun khuyến nghị cân nhắc giao dịch (mục 3.8, Đồ án tốt nghiệp).

Logic bám sát đúng 3 tiêu chí độc lập được mô tả trong đồ án, sau đó tổng hợp
trực tiếp thành nội dung nhận định / khuyến nghị cân nhắc giao dịch:

  1. Dấu của Spread dự báo  -> quan hệ giá vàng SJC vs. giá vàng thế giới quy đổi
  2. Vị trí của Spread dự báo so với tứ phân vị (Q1/Q2/Q3) của Spread lịch sử
  3. Mức độ biến động (Rolling Volatility, cửa sổ 30 ngày) so với tứ phân vị
     của Rolling Volatility lịch sử

Mô-đun KHÔNG tạo ra khuyến nghị mua/bán; chỉ tổng hợp 3 câu nhận định thành
nội dung tham khảo, đúng như phạm vi được nêu trong đồ án (mục 3.8.2.d).
"""

import numpy as np
import pandas as pd

ROLLING_WINDOW = 30  # cửa sổ Rolling Volatility theo đồ án (mục 3.8.2.c)


def _sign_text(value: float) -> str:
    if value > 0:
        return (
            "Spread dự báo mang giá trị dương, cho thấy giá vàng SJC cao hơn "
            "giá vàng thế giới quy đổi."
        )
    if value < 0:
        return (
            "Spread dự báo mang giá trị âm, cho thấy giá vàng SJC thấp hơn "
            "giá vàng thế giới quy đổi."
        )
    return "Spread dự báo bằng 0, cho thấy giá vàng SJC bằng giá vàng thế giới quy đổi."


def _position_text(value: float, q1: float, q2: float, q3: float) -> str:
    if value <= q1:
        return "Spread nằm trong khoảng giá trị thấp của dữ liệu lịch sử."
    if value <= q2:
        return "Spread nằm trong khoảng giá trị trung bình thấp của dữ liệu lịch sử."
    if value <= q3:
        return "Spread nằm trong khoảng giá trị trung bình cao của dữ liệu lịch sử."
    return "Spread nằm trong khoảng giá trị cao của dữ liệu lịch sử."


def _volatility_text(rv: float, q1: float, q2: float, q3: float) -> str:
    if pd.isna(rv):
        return "Chưa đủ dữ liệu lịch sử gần nhất để ước lượng mức biến động."
    if rv <= q1:
        return "Mức biến động của Spread nằm trong khoảng giá trị thấp của dữ liệu lịch sử."
    if rv <= q2:
        return (
            "Mức biến động của Spread nằm trong khoảng giá trị trung bình thấp "
            "của dữ liệu lịch sử."
        )
    if rv <= q3:
        return (
            "Mức biến động của Spread nằm trong khoảng giá trị trung bình cao "
            "của dữ liệu lịch sử."
        )
    return "Mức biến động của Spread nằm trong khoảng giá trị cao của dữ liệu lịch sử."


def generate_recommendations(
    forecast_values,
    historical_spread: pd.Series,
    window: int = ROLLING_WINDOW,
):
    """
    forecast_values: list/array các giá trị Spread dự báo, theo thứ tự thời gian
        (t+1, t+2, ..., t+N).
    historical_spread: pd.Series giá trị Spread thực tế trong quá khứ, đã sắp
        xếp theo thời gian tăng dần (không chứa NaN).

    Trả về list[dict], mỗi phần tử ứng với 1 ngày dự báo, gồm các nhận định
    thành phần và câu khuyến nghị tổng hợp.
    """

    hist = pd.Series(historical_spread).dropna().reset_index(drop=True)

    # Tứ phân vị của Spread lịch sử (mục 3.8.2.b)
    q1_s, q2_s, q3_s = hist.quantile([0.25, 0.5, 0.75])

    # Tứ phân vị của Rolling Volatility lịch sử (mục 3.8.2.c)
    hist_rv = hist.rolling(window).std()
    hist_rv = hist_rv.dropna()
    if len(hist_rv) > 0:
        q1_rv, q2_rv, q3_rv = hist_rv.quantile([0.25, 0.5, 0.75])
    else:
        q1_rv = q2_rv = q3_rv = np.nan

    # Cửa sổ trượt: bắt đầu bằng (window - 1) quan sát thực tế gần nhất,
    # sau đó lần lượt bổ sung từng giá trị Spread dự báo (đúng cách mô tả
    # trong Bảng 3.13 của đồ án).
    rolling_buffer = list(hist.tail(max(window - 1, 0)))

    results = []
    for day_idx, value in enumerate(forecast_values, start=1):
        rolling_buffer.append(value)
        window_vals = rolling_buffer[-window:]

        rv_t = np.std(window_vals, ddof=1) if len(window_vals) > 1 else np.nan

        sign_text = _sign_text(value)
        position_text = _position_text(value, q1_s, q2_s, q3_s)
        volatility_text = _volatility_text(rv_t, q1_rv, q2_rv, q3_rv)

        # 1. Gom các biến vào một danh sách
        raw_texts = [sign_text, position_text, volatility_text]

        # 2. Thêm dấu "- " vào trước mỗi đoạn text (chỉ lấy các biến có nội dung, bỏ qua biến rỗng)
        bullet_lines = [f"- {text.strip()}" for text in raw_texts if text and text.strip()]

        results.append(
            {
                "Ngày dự báo": f"t+{day_idx}",
                "Spread dự báo (VND/lượng)": f"{value:,.0f}",
                "Nhận định": "<br>".join(bullet_lines),
                "Rolling Volatility": rv_t,
                "Nhận định 1 - Dấu Spread": sign_text,
                "Nhận định 2 - Vị trí Spread": position_text,
                "Nhận định 3 - Mức biến động": volatility_text,
                "Khuyến nghị cân nhắc giao dịch": " ".join(
                    [sign_text, position_text, volatility_text]
                ),
            }
        )

    return results
