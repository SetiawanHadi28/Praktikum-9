import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, render_template, request  # noqa: E402
from sklearn.metrics import mean_squared_error  # noqa: E402


def load_dataset() -> pd.DataFrame:
    data_path = Path(__file__).parent / "covid_19_indonesia_time_series_all.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {data_path}. Pastikan file CSV tersedia."
        )

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    indo = df[df["Location"] == "Indonesia"]
    indo = indo[["Date", "Total Cases"]].dropna()
    return indo


def logistic_model(y: float, r: float, k: float) -> float:
    return r * y * (1 - y / k)


def euler_method(y0: float, r: float, k: float, h: float, n_steps: int) -> np.ndarray:
    y = np.zeros(n_steps)
    y[0] = y0
    for i in range(1, n_steps):
        y[i] = y[i - 1] + h * logistic_model(y[i - 1], r, k)
    return y


def build_plot(indo: pd.DataFrame, y_sim: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indo["Date"], indo["Total Cases"], label="Data Aktual")
    ax.plot(indo["Date"], y_sim, label="Simulasi Euler", linestyle="--")
    ax.set_title("Perbandingan Data Aktual vs Simulasi Euler")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Total Kasus")
    ax.legend()
    ax.grid(True)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


app = Flask(__name__)


@app.route("/")
def index():
    # Baca parameter dari query string; gunakan default jika tidak ada atau invalid
    def _as_float(value, default):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    r = _as_float(request.args.get("r"), 0.15)
    k_input = _as_float(request.args.get("k"), 0.0)
    h = _as_float(request.args.get("h"), 1.0)

    indo = load_dataset()
    y0 = float(indo["Total Cases"].iloc[0])
    default_k = float(indo["Total Cases"].max() * 1.2)
    k = k_input if k_input > 0 else default_k
    h = max(h, 0.01)  # step minimal
    n_steps = len(indo)

    y_sim = euler_method(y0, r, k, h, n_steps)
    mse = mean_squared_error(indo["Total Cases"], y_sim)
    plot_url = build_plot(indo, y_sim)
    preview_rows = []
    for _, row in indo.head(10).iterrows():
        preview_rows.append(
            {
                "date": row["Date"].strftime("%Y-%m-%d"),
                "total_cases": f"{int(row['Total Cases']):,}",
            }
        )

    return render_template(
        "index.html",
        mse=mse,
        plot_url=plot_url,
        preview_rows=preview_rows,
        total_rows=len(indo),
        r=r,
        k=k,
        h=h,
        default_k=round(default_k, 2),
    )


if __name__ == "__main__":
    app.run(debug=True)

