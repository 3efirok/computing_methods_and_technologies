import os
import ssl
import urllib.request
import json
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Лабораторна робота №1: Інтерполяція кубічними сплайнами
# Маршрут: т/б "Заросляк" — вершина гори Говерла
# Виконав: Накладюк Володимир
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Отримання геоданих через Open-Elevation API (або резервні дані)
# ------------------------------------------------------------------------------
api_url = (
    "https://api.open-elevation.com/api/v1/lookup?locations="
    "48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|"
    "48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|"
    "48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|"
    "48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|"
    "48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|"
    "48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|"
    "48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
)

fallback_results = [
    {"latitude": 48.164214, "longitude": 24.536044, "elevation": 1264.0},
    {"latitude": 48.164983, "longitude": 24.534836, "elevation": 1285.0},
    {"latitude": 48.165605, "longitude": 24.534068, "elevation": 1285.0},
    {"latitude": 48.166228, "longitude": 24.532915, "elevation": 1333.0},
    {"latitude": 48.166777, "longitude": 24.531927, "elevation": 1310.0},
    {"latitude": 48.167326, "longitude": 24.530884, "elevation": 1318.0},
    {"latitude": 48.167011, "longitude": 24.530061, "elevation": 1318.0},
    {"latitude": 48.166053, "longitude": 24.528039, "elevation": 1339.0},
    {"latitude": 48.166655, "longitude": 24.526064, "elevation": 1375.0},
    {"latitude": 48.166497, "longitude": 24.523574, "elevation": 1417.0},
    {"latitude": 48.166128, "longitude": 24.520214, "elevation": 1486.0},
    {"latitude": 48.165416, "longitude": 24.517170, "elevation": 1524.0},
    {"latitude": 48.164546, "longitude": 24.514640, "elevation": 1553.0},
    {"latitude": 48.163412, "longitude": 24.512980, "elevation": 1630.0},
    {"latitude": 48.162331, "longitude": 24.511715, "elevation": 1757.0},
    {"latitude": 48.162015, "longitude": 24.509462, "elevation": 1794.0},
    {"latitude": 48.162147, "longitude": 24.506932, "elevation": 1828.0},
    {"latitude": 48.161751, "longitude": 24.504244, "elevation": 1887.0},
    {"latitude": 48.161197, "longitude": 24.501793, "elevation": 1975.0},
    {"latitude": 48.160580, "longitude": 24.500537, "elevation": 1975.0},
    {"latitude": 48.160250, "longitude": 24.500106, "elevation": 2031.0}
]

try:
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
        data = json.loads(resp.read().decode())
        results = data["results"]
        print("Дані успішно отримано через Open-Elevation API.")
except Exception as err:
    print(f"API недоступне в поточному середовищі ({err}). Використовуються збережені опорні дані маршруту.")
    results = fallback_results

# ------------------------------------------------------------------------------
# 2. Табуляція вузлів та запис у текстовий файл (п. 2, 3 методички)
# ------------------------------------------------------------------------------
num_points = len(results)
print(f"\nКількість вузлів: {num_points}")
print("№  | Latitude  | Longitude | Elevation (m)")
print("-" * 44)
for i, point in enumerate(results):
    print(f"{i:2d} | {point['latitude']:.6f} | {point['longitude']:.6f} | {point['elevation']:.2f}")

with open("tabulation.txt", "w", encoding="utf-8") as f:
    f.write(f"Табуляція координат маршруту Заросляк - Говерла (Вузлів: {num_points})\n")
    f.write(f"{'№':<3} | {'Latitude':<11} | {'Longitude':<11} | {'Elevation (m)':<13}\n")
    f.write("-" * 47 + "\n")
    for i, p in enumerate(results):
        f.write(f"{i:<3d} | {p['latitude']:<11.6f} | {p['longitude']:<11.6f} | {p['elevation']:<13.2f}\n")

# ------------------------------------------------------------------------------
# 3. Обчислення кумулятивної відстані (п. 4 методички)
# ------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Обчислення ортодромічної відстані між двома точками за формулою гаверсинуса"""
    R = 6371000.0  # Радіус Землі в метрах
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return 2.0 * R * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]

distances = [0.0]
for i in range(1, num_points):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

print("\nТабуляція (відстань, висота):")
print("№  | Distance (m) | Elevation (m)")
print("-" * 34)
for i in range(num_points):
    print(f"{i:2d} | {distances[i]:12.2f} | {elevations[i]:13.2f}")

# ------------------------------------------------------------------------------
# 4. Метод кубічних сплайнів та метод прогонки (п. 6, 7, 8, 9 методички)
# ------------------------------------------------------------------------------
def build_cubic_spline(x_nodes, y_nodes):
    """
    Побудова природного кубічного сплайна:
    S_i(x) = a_i + b_i*(x - x_{i-1}) + c_i*(x - x_{i-1})^2 + d_i*(x - x_{i-1})^3,
    для i = 1, ..., n на відрізках [x_{i-1}, x_i].
    """
    n = len(x_nodes) - 1
    x = np.array(x_nodes, dtype=float)
    y = np.array(y_nodes, dtype=float)

    # Кроки h_i = x_i - x_{i-1} для i = 1..n
    h = np.zeros(n + 1)
    for i in range(1, n + 1):
        h[i] = x[i] - x[i - 1]

    # Формування тридіагональної СЛАР для коефіцієнтів c_i
    alpha = np.zeros(n + 1)
    beta = np.zeros(n + 1)
    gamma = np.zeros(n + 1)
    delta = np.zeros(n + 1)

    # Крайова умова при x = x_0: c_1 = 0
    beta[1] = 1.0
    gamma[1] = 0.0
    delta[1] = 0.0

    # Внутрішні вузли i = 2, ..., n-1
    for i in range(2, n):
        alpha[i] = h[i - 1]
        beta[i] = 2.0 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3.0 * ((y[i] - y[i - 1]) / h[i] - (y[i - 1] - y[i - 2]) / h[i - 1])

    # Останній вузол i = n
    alpha[n] = h[n - 1]
    beta[n] = 2.0 * (h[n - 1] + h[n])
    gamma[n] = 0.0
    delta[n] = 3.0 * ((y[n] - y[n - 1]) / h[n] - (y[n - 1] - y[n - 2]) / h[n - 1])

    # Метод прогонки: прямий хід (коефіцієнти A_i, B_i)
    A = np.zeros(n + 1)
    B = np.zeros(n + 1)

    A[1] = -gamma[1] / beta[1]
    B[1] = delta[1] / beta[1]

    for i in range(2, n):
        denom = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    # Зворотний хід: знаходження c_n, c_{n-1}, ..., c_1
    c = np.zeros(n + 1)
    c[n] = (delta[n] - alpha[n] * B[n - 1]) / (alpha[n] * A[n - 1] + beta[n])
    for i in range(n - 1, 0, -1):
        c[i] = A[i] * c[i + 1] + B[i]

    # Обчислення коефіцієнтів a_i, b_i, d_i
    a = np.zeros(n + 1)
    b = np.zeros(n + 1)
    d = np.zeros(n + 1)

    for i in range(1, n + 1):
        a[i] = y[i - 1]

    for i in range(1, n):
        d[i] = (c[i + 1] - c[i]) / (3.0 * h[i])
        b[i] = (y[i] - y[i - 1]) / h[i] - (h[i] / 3.0) * (c[i + 1] + 2.0 * c[i])

    d[n] = -c[n] / (3.0 * h[n])
    b[n] = (y[n] - y[n - 1]) / h[n] - (2.0 / 3.0) * h[n] * c[n]

    # Перевірка нев'язки системи (Residual check)
    residuals = np.zeros(n + 1)
    residuals[1] = abs(beta[1] * c[1] + gamma[1] * c[2] - delta[1]) if n >= 2 else 0
    for i in range(2, n):
        residuals[i] = abs(alpha[i] * c[i - 1] + beta[i] * c[i] + gamma[i] * c[i + 1] - delta[i])
    residuals[n] = abs(alpha[n] * c[n - 1] + beta[n] * c[n] - delta[n])
    max_residual = np.max(residuals[1:])

    return {
        "n": n, "x": x, "y": y, "h": h,
        "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
        "A": A, "B": B, "a": a, "b": b, "c": c, "d": d,
        "max_residual": max_residual
    }

def evaluate_spline(spline, x_eval):
    """Обчислення значень кубічного сплайна в заданих точках"""
    x_arr = np.asarray(x_eval)
    y_eval = np.zeros_like(x_arr, dtype=float)
    x = spline["x"]
    n = spline["n"]
    a, b, c, d = spline["a"], spline["b"], spline["c"], spline["d"]

    for idx, val in np.ndenumerate(x_arr):
        if val <= x[0]:
            i = 1
        elif val >= x[-1]:
            i = n
        else:
            i = int(np.searchsorted(x, val))
            if i == 0:
                i = 1
        dx = val - x[i - 1]
        y_eval[idx] = a[i] + b[i] * dx + c[i] * (dx**2) + d[i] * (dx**3)
    return y_eval

def evaluate_spline_derivative(spline, x_eval):
    """Обчислення аналітичної похідної кубічного сплайна S'(x)"""
    x_arr = np.asarray(x_eval)
    deriv = np.zeros_like(x_arr, dtype=float)
    x = spline["x"]
    n = spline["n"]
    b, c, d = spline["b"], spline["c"], spline["d"]

    for idx, val in np.ndenumerate(x_arr):
        if val <= x[0]:
            i = 1
        elif val >= x[-1]:
            i = n
        else:
            i = int(np.searchsorted(x, val))
            if i == 0:
                i = 1
        dx = val - x[i - 1]
        deriv[idx] = b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * (dx**2)
    return deriv

# Побудова повного сплайна за всіма 21 точками
full_spline = build_cubic_spline(distances, elevations)
n_intervals = full_spline["n"]

print("\nКоефіцієнти тридіагональної системи (СЛАР):")
print(" i |    alpha_i    |     beta_i    |    gamma_i    |    delta_i")
print("-" * 62)
for i in range(1, n_intervals + 1):
    print(f"{i:2d} | {full_spline['alpha'][i]:13.3f} | {full_spline['beta'][i]:13.3f} | "
          f"{full_spline['gamma'][i]:13.3f} | {full_spline['delta'][i]:13.6f}")

print(f"\nМаксимальна нев'язка розв'язку СЛАР методом прогонки: {full_spline['max_residual']:.2e}")

print("\nКоефіцієнти кубічних сплайнів S_i(x):")
print(" i |      a_i     |      b_i     |      c_i     |      d_i")
print("-" * 62)
for i in range(1, n_intervals + 1):
    print(f"{i:2d} | {full_spline['a'][i]:12.3f} | {full_spline['b'][i]:12.5f} | "
          f"{full_spline['c'][i]:12.6f} | {full_spline['d'][i]:12.8f}")

# ------------------------------------------------------------------------------
# 5. Графіки та дослідження (п. 5, 10, 11, 12 методички)
# ------------------------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

# Графік 1: Дискретний профіль висоти (п. 5)
plt.figure(figsize=(9, 5))
plt.scatter(distances, elevations, color="#d9534f", s=45, zorder=3, label="GPS-точки маршруту")
plt.plot(distances, elevations, color="#0275d8", linestyle="--", alpha=0.6, label="Ламана лінія")
plt.title("Дискретний висотний профіль маршруту (Заросляк — Говерла)", fontsize=12)
plt.xlabel("Кумулятивна відстань (м)", fontsize=11)
plt.ylabel("Висота над рівнем моря (м)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("plots/profile_discrete.png", dpi=300)
plt.close()

# Дослідження впливу кількості вузлів (10, 15, 20) на точність (п. 10, 11)
node_counts = [10, 15, 20]
spline_variants = {}
error_metrics = {}

full_x = np.array(distances)
full_y = np.array(elevations)
xx_dense = np.linspace(0, distances[-1], 1000)

for k in node_counts:
    indices = np.unique(np.round(np.linspace(0, len(full_x) - 1, k)).astype(int))
    sub_x = full_x[indices]
    sub_y = full_y[indices]
    sp = build_cubic_spline(sub_x, sub_y)
    spline_variants[k] = (sub_x, sub_y, sp)

    # Оцінка похибки у всіх точках початкового набору
    y_pred = evaluate_spline(sp, full_x)
    abs_err = np.abs(full_y - y_pred)
    error_metrics[k] = {
        "max_err": np.max(abs_err),
        "mean_err": np.mean(abs_err),
        "rmse": np.sqrt(np.mean(abs_err**2)),
        "y_pred": y_pred,
        "abs_err": abs_err
    }

print("\nПорівняння точності інтерполяції залежно від кількості вузлів:")
print("Вузли | Max Error (м) | Mean Error (м) |  RMSE (м)")
print("-" * 46)
for k in node_counts:
    m = error_metrics[k]
    print(f" {k:2d}   |    {m['max_err']:9.2f}  |    {m['mean_err']:10.2f}  |  {m['rmse']:7.2f}")

# Графік 2: Порівняння сплайнів для 10, 15 та 20 вузлів (п. 10)
plt.figure(figsize=(10, 5.5))
colors = {10: "#f0ad4e", 15: "#5bc0de", 20: "#5cb85c"}
for k in node_counts:
    sub_x, sub_y, sp = spline_variants[k]
    yy_curve = evaluate_spline(sp, xx_dense)
    plt.plot(xx_dense, yy_curve, label=f"Сплайн ({k} вузлів)", color=colors[k], linewidth=1.8)
    plt.scatter(sub_x, sub_y, color=colors[k], s=25, alpha=0.8)

plt.scatter(distances, elevations, color="black", s=35, zorder=5, label="Оригінальні GPS-точки (21)")
plt.title("Інтерполяція кубічними сплайнами для різної кількості вузлів", fontsize=12)
plt.xlabel("Кумулятивна відстань (м)", fontsize=11)
plt.ylabel("Висота (м)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("plots/splines_comparison.png", dpi=300)
plt.close()

# Графік 3: Задана функція, наближення та похибка (п. 12)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

yy_full = evaluate_spline(full_spline, xx_dense)
ax1.plot(xx_dense, yy_full, color="#0275d8", linewidth=2, label="Сплайн-апроксимація y_набл = S(x)")
ax1.scatter(distances, elevations, color="#d9534f", s=35, zorder=4, label="Фактичні вузли рельєфу y = f(x)")
ax1.set_ylabel("Висота (м)", fontsize=11)
ax1.set_title("Наближення кубічним сплайном висотного профілю та графік похибки", fontsize=12)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc="upper left")

ax2.plot(full_x, error_metrics[10]["abs_err"], color="#f0ad4e", marker="o", markersize=4,
         label="Похибка epsilon для 10 вузлів")
ax2.plot(full_x, error_metrics[15]["abs_err"], color="#5bc0de", marker="s", markersize=4,
         label="Похибка epsilon для 15 вузлів")
ax2.plot(full_x, error_metrics[20]["abs_err"], color="#5cb85c", marker="^", markersize=4,
         label="Похибка epsilon для 20 вузлів")
ax2.set_xlabel("Кумулятивна відстань (м)", fontsize=11)
ax2.set_ylabel("Похибка |y - y_набл| (м)", fontsize=11)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig("plots/approximation_error.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 6. Додаткові характеристики маршруту та градієнтний аналіз
# ------------------------------------------------------------------------------
total_length = distances[-1]
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0.0) for i in range(1, num_points))
total_descent = sum(max(elevations[i - 1] - elevations[i], 0.0) for i in range(1, num_points))

print("\n" + "=" * 50)
print("ХАРАКТЕРИСТИКИ МАРШРУТУ:")
print(f"Загальна довжина маршруту: {total_length:.2f} м ({total_length / 1000.0:.2f} км)")
print(f"Сумарний набір висоти:    {total_ascent:.2f} м")
print(f"Сумарний спуск:           {total_descent:.2f} м")

grad_full = evaluate_spline_derivative(full_spline, xx_dense) * 100.0  # У відсотках (%)
max_ascent = np.max(grad_full)
max_descent = np.min(grad_full)
mean_gradient = np.mean(np.abs(grad_full))

dx_step = xx_dense[1] - xx_dense[0]
steep_sections_mask = np.abs(grad_full) > 15.0
steep_length = np.sum(steep_sections_mask) * dx_step
steep_percent = (steep_length / total_length) * 100.0

print(f"Максимальний підйом:      {max_ascent:.2f} %")
print(f"Максимальний спуск:       {max_descent:.2f} %")
print(f"Середній градієнт:        {mean_gradient:.2f} %")
print(f"Ділянки з крутизною > 15%: {steep_length:.2f} м ({steep_percent:.1f} % маршруту)")

# Графік 4: Профіль ухилу (градієнта)
plt.figure(figsize=(10, 4.8))
plt.plot(xx_dense, grad_full, color="#6f42c1", linewidth=1.8, label="Градієнт маршруту S'(x) (%)")
plt.axhline(15, color="red", linestyle="--", linewidth=1, label="Поріг крутизни 15%")
plt.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
plt.fill_between(xx_dense, grad_full, 15, where=(grad_full >= 15), color="red", alpha=0.2,
                 label="Круті ділянки (> 15%)")
plt.title("Профіль крутизни (градієнта) маршруту Заросляк — Говерла", fontsize=12)
plt.xlabel("Кумулятивна відстань (м)", fontsize=11)
plt.ylabel("Ухил (%)", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("plots/gradient_profile.png", dpi=300)
plt.close()

# Механічна енергія підйому для m = 80 кг
mass = 80.0
g = 9.81
energy_j = mass * g * total_ascent
energy_kj = energy_j / 1000.0
energy_kcal = energy_j / 4184.0

print(f"Механічна робота:         {energy_j:.2f} Дж")
print(f"Механічна робота:         {energy_kj:.2f} кДж")
print(f"Енергія:                  {energy_kcal:.2f} ккал")
print("=" * 50)
print("Обчислення завершено успішно. Графіки збережено у папку plots/.")

