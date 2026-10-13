"""
Лабораторна робота №3
Знаходження алгебраїчних многочленів найкращого квадратичного наближення
методом найменших квадратів (МНК).
Виконав: Накладюк Володимир
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def read_data(filepath):
    """Зчитування вхідних даних з CSV-файлу."""
    months = []
    temps = []
    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # Пропуск заголовка
        for row in reader:
            if row:
                months.append(float(row[0]))
                temps.append(float(row[1]))
    return np.array(months), np.array(temps)


def form_matrix(x, m):
    """Формування матриці B нормальної системи розміром (m+1, m+1)."""
    B = np.zeros((m + 1, m + 1), dtype=float)
    for k in range(m + 1):
        for j in range(m + 1):
            B[k, j] = np.sum(x ** (k + j))
    return B


def form_vector(x, y, m):
    """Формування вектора C вільних членів розміром m+1."""
    C = np.zeros(m + 1, dtype=float)
    for k in range(m + 1):
        C[k] = np.sum(y * (x ** k))
    return C


def gauss_solve(A, b):
    """
    Розв'язання СЛАР методом Гауса з вибором головного елемента по стовпцях.
    A - матриця системи (n, n), b - вектор правих частин (n).
    """
    A = A.copy().astype(float)
    b = b.copy().astype(float)
    n = len(b)

    # Прямий хід
    for k in range(n - 1):
        max_row = k + int(np.argmax(np.abs(A[k:, k])))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]

        if np.abs(A[k, k]) < 1e-15:
            continue

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід
    x_sol = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i + 1:] * x_sol[i + 1:])) / A[i, i]

    return x_sol


def polynomial(x, coef):
    """Обчислення значення многочлена phi(x) = sum(a_j * x^j)."""
    y_poly = np.zeros_like(x, dtype=float)
    for i, a in enumerate(coef):
        y_poly += a * (x ** i)
    return y_poly


def variance(y_true, y_approx):
    """Обчислення дисперсії відхилень (середньоквадратичної помилки)."""
    return float(np.mean((y_true - y_approx) ** 2))


def compute_error(y_true, y_approx):
    """Обчислення модуля похибки epsilon(x) = |f(x) - phi(x)|."""
    return np.abs(y_true - y_approx)


def tabulate_errors(x, y, coefs_dict, output_file="tabulation_errors.txt"):
    """
    Табулювання похибки на відрізку [x_0, x_n] з кроком h1 = (x_n - x_0) / (20 * n).
    """
    x0, xn = x[0], x[-1]
    n = len(x) - 1
    h1 = (xn - x0) / (20.0 * n)
    dense_x = np.linspace(x0, xn, 20 * n + 1)

    f_dense = np.interp(dense_x, x, y)

    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, mode="w", encoding="utf-8") as f:
        header_degrees = [f"m={m}" for m in range(1, 11)]
        f.write(f"Табулювання похибок апроксимації на відрізку [{x0:.1f}, {xn:.1f}] з кроком h1 = {h1:.4f}\n")
        f.write("=" * 120 + "\n")
        f.write(f"{'x':>8} | {'f(x)':>8} | " + " | ".join(f"{deg:>9}" for deg in header_degrees) + "\n")
        f.write("-" * 120 + "\n")

        for xi, fi in zip(dense_x, f_dense):
            row_errors = []
            for m in range(1, 11):
                y_pred = polynomial(xi, coefs_dict[m])
                err = abs(fi - y_pred)
                row_errors.append(f"{err:9.4f}")
            f.write(f"{xi:8.4f} | {fi:8.2f} | " + " | ".join(row_errors) + "\n")


def plot_results(x, y, coefs_dict, variances, optimal_m):
    """Побудова та збереження графіків (300 DPI)."""
    os.makedirs("plots", exist_ok=True)
    degrees = list(range(1, 11))
    std_devs = [np.sqrt(v) for v in variances]

    # 1. Графік залежності дисперсії та СКВ від степеня m
    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(degrees, variances, marker="o", color="#1f77b4", linewidth=2, label="Дисперсія $\\sigma^2$")
    plt.plot(degrees, std_devs, marker="s", color="#ff7f0e", linewidth=2, linestyle="--", label="СКВ $\\delta$ (°C)")
    plt.axvline(optimal_m, color="#2ca02c", linestyle=":", label=f"Оптимальний ступінь m={optimal_m}")
    plt.title("Залежність дисперсії та похибки від степеня многочлена $m$", fontsize=13, fontweight="bold")
    plt.xlabel("Ступінь многочлена $m$", fontsize=11)
    plt.ylabel("Величина дисперсії / СКВ", fontsize=11)
    plt.xticks(degrees)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/dispersion_vs_degree.png")
    plt.close()

    # 2. Графік фактичних даних та апроксимації (m = 1, 2, 4, optimal_m)
    x_dense = np.linspace(1, 24, 400)
    plt.figure(figsize=(11, 6), dpi=300)
    plt.plot(x, y, "ko", markersize=6, label="Фактичні спостереження $y_i$")
    plt.plot(x, y, "k--", alpha=0.3, label="Лінійне з'єднання вузлів")

    for m, col, ls in [(1, "#9467bd", ":"), (2, "#d62728", "-."), (4, "#2ca02c", "-"), (6, "#1f77b4", "-")]:
        if m in coefs_dict:
            lbl = f"Поліном m={m}" + (" (оптимальний)" if m == optimal_m else "")
            plt.plot(x_dense, polynomial(x_dense, coefs_dict[m]), color=col, linestyle=ls, linewidth=2, label=lbl)

    plt.title("Апроксимація температурних даних многочленами МНК", fontsize=13, fontweight="bold")
    plt.xlabel("Місяць", fontsize=11)
    plt.ylabel("Температура (°C)", fontsize=11)
    plt.xticks(np.arange(1, 25))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/approximation_comparison.png")
    plt.close()

    # 3. Графік похибок апроксимації epsilon(x)
    plt.figure(figsize=(11, 5), dpi=300)
    f_dense = np.interp(x_dense, x, y)
    for m, col, ls in [(1, "#9467bd", ":"), (2, "#d62728", "-."), (4, "#2ca02c", "-"), (6, "#1f77b4", "-")]:
        if m in coefs_dict:
            err = np.abs(f_dense - polynomial(x_dense, coefs_dict[m]))
            plt.plot(x_dense, err, color=col, linestyle=ls, linewidth=1.8, label=f"Похибка $\\varepsilon(x)$, m={m}")

    plt.title("Похибка апроксимації $\\varepsilon(x) = |f(x) - \\varphi(x)|$ на відрізку [1, 24]", fontsize=13, fontweight="bold")
    plt.xlabel("Місяць", fontsize=11)
    plt.ylabel("Абсолютна похибка (°C)", fontsize=11)
    plt.xticks(np.arange(1, 25))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/approximation_errors.png")
    plt.close()

    # 4. Графік екстраполяції та прогнозу на місяці 25, 26, 27
    x_extrap = np.linspace(1, 27.5, 500)
    future_months = np.array([25, 26, 27])

    plt.figure(figsize=(11, 6), dpi=300)
    plt.plot(x, y, "ko", markersize=6, label="Фактичні дані (місяці 1-24)")
    plt.axvspan(24, 27.5, color="#ffecb3", alpha=0.4, label="Зона прогнозу (місяці 25-27)")

    for m, col, ls in [(2, "#d62728", "-."), (4, "#2ca02c", "-"), (6, "#1f77b4", "--")]:
        y_extrap = polynomial(x_extrap, coefs_dict[m])
        plt.plot(x_extrap, y_extrap, color=col, linestyle=ls, linewidth=2, label=f"Многочлен m={m}")
        preds = polynomial(future_months, coefs_dict[m])
        plt.plot(future_months, preds, marker="o", color=col, linestyle="none")
        for xm, ym in zip(future_months, preds):
            if -30 <= ym <= 40:
                plt.annotate(f"{ym:.1f}°", (xm, ym), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    plt.ylim(-40, 35)
    plt.title("Екстраполяція та прогноз температури на місяці 25, 26, 27", fontsize=13, fontweight="bold")
    plt.xlabel("Місяць", fontsize=11)
    plt.ylabel("Температура (°C)", fontsize=11)
    plt.xticks(np.arange(1, 28))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/forecast_extrapolation.png")
    plt.close()


def main():
    data_file = "temperature.csv"
    if not os.path.exists(data_file):
        print(f"Помилка: файл {data_file} не знайдено!")
        return

    # 1. Зчитування вхідних даних
    x, y = read_data(data_file)
    n = len(x) - 1

    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №3: МЕТОД НАЙМЕНШИХ КВАДРАТІВ (МНК)")
    print(f"Завантажено {len(x)} спостережень з {data_file}")
    print("=" * 80)

    # 2. Розрахунок многочленів степенів m = 1..10
    coefs_dict = {}
    variances = []

    print("\nТаблиця розрахунку дисперсій для степенів m = 1 .. 10:")
    print("-" * 65)
    print(f"{'m':>3} | {'Дисперсія (var)':>16} | {'СКВ delta (°C)':>15} | {'Резюме':>20}")
    print("-" * 65)

    for m in range(1, 11):
        B = form_matrix(x, m)
        C = form_vector(x, y, m)
        coef = gauss_solve(B, C)
        coefs_dict[m] = coef

        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        delta = np.sqrt(var)
        variances.append(var)

        note = ""
        if m == 1:
            note = "лінійний тренд"
        elif m == 2:
            note = "парабола"
        elif m == 4:
            note = "якісний перелом"
        elif m == 6:
            note = "висока точність"

        print(f"{m:3d} | {var:16.4f} | {delta:15.4f} | {note:>20}")

    optimal_m = 4
    print("-" * 65)
    print(f"Оптимальний ступінь многочлена за методичкою (max_degree=4): m = {optimal_m}")
    print(f"Дисперсія sigma^2 = {variances[optimal_m - 1]:.4f}, СКВ delta = {np.sqrt(variances[optimal_m - 1]):.4f} °C")

    # Коефіцієнти оптимального многочлена
    opt_coef = coefs_dict[optimal_m]
    print("\nКоефіцієнти оптимального многочлена m = 4:")
    for j, aj in enumerate(opt_coef):
        print(f"  a{j} = {aj:+.6e}")

    # 3. Прогноз на місяці 25, 26, 27
    future_months = np.array([25, 26, 27], dtype=float)
    print("\nПрогноз середньомісячної температури на наступні 3 місяці (25, 26, 27):")
    print("-" * 65)
    print(f"{'Ступінь m':>10} | {'Місяць 25':>12} | {'Місяць 26':>12} | {'Місяць 27':>12}")
    print("-" * 65)
    for m in [1, 2, 4, 6]:
        preds = polynomial(future_months, coefs_dict[m])
        lbl = f"m = {m}" + (" (опт.)" if m == optimal_m else "")
        print(f"{lbl:>10} | {preds[0]:12.2f} | {preds[1]:12.2f} | {preds[2]:12.2f}")
    print("-" * 65)

    # 4. Табулювання похибок
    tabulation_file = "tabulation_errors.txt"
    tabulate_errors(x, y, coefs_dict, tabulation_file)
    print(f"\n[+] Результати табулювання похибок збережено у файл: {tabulation_file}")

    # 5. Побудова графіків
    plot_results(x, y, coefs_dict, variances, optimal_m)
    print("[+] Графіки успішно збережено у директорію 'plots/':")
    print("    - plots/dispersion_vs_degree.png")
    print("    - plots/approximation_comparison.png")
    print("    - plots/approximation_errors.png")
    print("    - plots/forecast_extrapolation.png")
    print("\nРоботу програми успішно завершено.")


if __name__ == "__main__":
    main()

