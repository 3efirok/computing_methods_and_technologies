"""
Лабораторна робота №4
Точність формул для чисельного диференціювання.
Метод Рунге-Ромберга. Метод Ейткена.
Виконав: Накладюк Володимир
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def M(t):
    """Функція вологості ґрунту M(t) = 50*exp(-0.1*t) + 5*sin(t)."""
    return 50.0 * np.exp(-0.1 * t) + 5.0 * np.sin(t)


def dM_exact(t):
    """Точна аналітична похідна dM/dt = -5*exp(-0.1*t) + 5*cos(t)."""
    return -5.0 * np.exp(-0.1 * t) + 5.0 * np.cos(t)


def diff_central(f, t, h):
    """
    Центральна різницева формула для першої похідної:
    y'_0(h) = (f(t + h) - f(t - h)) / (2*h). Порядок точності O(h^2).
    """
    return (f(t + h) - f(t - h)) / (2.0 * h)


def runge_romberg(d_h, d_2h, p=2):
    """
    Уточнення похідної за методом Рунге-Ромберга:
    y'_R = y'_0(h) + (y'_0(h) - y'_0(2h)) / (2^p - 1).
    Для центральної різниці p = 2, дільник = 3.
    """
    return d_h + (d_h - d_2h) / (2.0**p - 1.0)


def aitken(d_h, d_2h, d_4h):
    """
    Уточнення за методом Ейткена та оцінка порядку точності:
    y'_E = (y'_0(2h)^2 - y'_0(4h)*y'_0(h)) / (2*y'_0(2h) - (y'_0(4h) + y'_0(h)))
    p = (1 / ln(2)) * ln(|(y'_0(4h) - y'_0(2h)) / (y'_0(2h) - y'_0(h))|)
    """
    numerator = d_2h**2 - d_4h * d_h
    denominator = 2.0 * d_2h - (d_4h + d_h)
    d_aitken = numerator / denominator

    # Оцінка фактичного порядку точності p
    ratio = abs((d_4h - d_2h) / (d_2h - d_h))
    p_order = np.log(ratio) / np.log(2.0)

    return d_aitken, p_order


def plot_results(h_grid, err_grid, h_opt, r_opt, h_base, err_h, err_2h, err_4h, err_rr, err_aitken, t0, exact_val):
    """Побудова та збереження графіків (300 DPI)."""
    os.makedirs("plots", exist_ok=True)

    # 1. Графік регуляризації: залежність похибки від кроку h (log-log)
    plt.figure(figsize=(9, 5.5), dpi=300)
    plt.loglog(h_grid, err_grid, label="Повна похибка $R(h) = |y'_0(h) - M'(t_0)|$", color="#1f77b4", linewidth=2)

    # Оптимальний крок h_opt
    plt.scatter([h_opt], [r_opt], color="#d62728", s=90, zorder=5,
                label=f"Оптимальний крок $h_0 \\approx {h_opt:.2e}$ ($R_0 \\approx {r_opt:.2e}$)")

    # Робочі точки h, 2h, 4h
    plt.scatter([h_base, 2 * h_base, 4 * h_base], [err_h, err_2h, err_4h],
                color="#2ca02c", s=70, zorder=5, marker="s",
                label=f"Робочі точки $h=10^{{-3}}, 2h, 4h$ ($O(h^2)$)")

    plt.axvline(h_opt, color="#d62728", linestyle="--", alpha=0.5)
    plt.text(1e-18, 1e-3, "Зона похибки округлення\n$\\sim \\varepsilon_{mach} / h$ (нахил -1)",
             fontsize=10, color="#8c564b", bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#8c564b", lw=1))
    plt.text(5e-2, 1e-12, "Зона похибки методу\n$\\sim O(h^2)$ (нахил +2)",
             fontsize=10, color="#1f77b4", bbox=dict(boxstyle="round,pad=0.3", fc="#eaf2f8", ec="#1f77b4", lw=1))

    plt.title("Дослідження похибки чисельного диференціювання від кроку $h$ (регуляризація)", fontsize=12, pad=10)
    plt.xlabel("Крок сітки $h$ (лог. масштаб)", fontsize=11)
    plt.ylabel("Абсолютна похибка $R(h)$ (лог. масштаб)", fontsize=11)
    plt.ylim(1e-15, 1e2)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="upper center", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("plots/regularization_curve.png")
    plt.close()

    # 2. Порівняння похибок методів на робочому кроці h = 10^-3
    methods = [
        "ЦР ($4h$)\n$h=4\\cdot 10^{-3}$",
        "ЦР ($2h$)\n$h=2\\cdot 10^{-3}$",
        "ЦР ($h$)\n$h=10^{-3}$",
        "Рунге-Ромберг\n($y'_R$)",
        "Ейткен\n($y'_E$)"
    ]
    errors = [err_4h, err_2h, err_h, err_rr, err_aitken]
    colors = ["#aec7e8", "#7bafde", "#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(8.5, 5), dpi=300)
    bars = plt.bar(methods, errors, color=colors, width=0.55, edgecolor="#333333", linewidth=1)
    plt.yscale("log")
    plt.title("Порівняння абсолютної похибки методів чисельного диференціювання", fontsize=12, pad=10)
    plt.ylabel("Абсолютна похибка $|y' - M'(t_0)|$ (лог. масштаб)", fontsize=11)
    plt.grid(axis="y", ls=":", alpha=0.7)

    for bar, err in zip(bars, errors):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.5,
                 f"{err:.2e}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(1e-13, 1e-4)
    plt.tight_layout()
    plt.savefig("plots/methods_comparison.png")
    plt.close()

    # 3. Графік функції вологості M(t) та швидкості зміни dM/dt
    t_vals = np.linspace(0, 5, 500)
    m_vals = M(t_vals)
    dm_vals = dM_exact(t_vals)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    color1 = "#1f77b4"
    ax1.set_xlabel("Час $t$ (умовні одиниці)", fontsize=11)
    ax1.set_ylabel("Вологість ґрунту $M(t)$ (%)", color=color1, fontsize=11)
    l1 = ax1.plot(t_vals, m_vals, color=color1, linewidth=2, label="Вологість $M(t)$")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, ls=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Швидкість зміни вологості $M'(t)$ (%/од. часу)", color=color2, fontsize=11)
    l2 = ax2.plot(t_vals, dm_vals, color=color2, linewidth=2, linestyle="--", label="Похідна $M'(t)$")
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.scatter([t0], [M(t0)], color="#2ca02c", s=70, zorder=5)
    ax2.scatter([t0], [exact_val], color="#2ca02c", s=70, zorder=5)
    ax2.annotate(f"$t_0=1.0$\n$M'(1)={exact_val:.4f}$",
                 xy=(t0, exact_val), xytext=(t0 + 0.3, exact_val - 1.5),
                 arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                 fontsize=9.5, bbox=dict(boxstyle="round,pad=0.2", fc="#ffffcc", ec="#999999"))

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", fontsize=10)
    plt.title("Динаміка вологості ґрунту $M(t)$ та її першої похідної $M'(t)$", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig("plots/moisture_dynamics.png")
    plt.close()


def main():
    t0 = 1.0
    print("=" * 78)
    print("ЛАБОРАТОРНА РОБОТА №4: Чисельне диференціювання, Рунге-Ромберг, Ейткен")
    print("=" * 78)

    # 1. Аналітичне значення похідної
    exact_val = dM_exact(t0)
    print(f"\n1. Точне аналітичне значення похідної M'(t) при t0 = {t0}:")
    print(f"   M'({t0}) = -5*exp(-0.1) + 5*cos(1) = {exact_val:.12f}")

    # 2. Дослідження залежності похибки від кроку h у діапазоні [10^-20 ... 10^3]
    h_powers = np.linspace(-20, 3, 2301)
    h_grid = 10.0**h_powers
    d_grid = diff_central(M, t0, h_grid)
    err_grid = np.abs(d_grid - exact_val)

    min_idx = int(np.argmin(err_grid))
    h_opt = float(h_grid[min_idx])
    r_opt = float(err_grid[min_idx])
    print("\n2. Дослідження похибки на сітці h in [10^-20 ... 10^3]:")
    print(f"   Оптимальний крок h0: {h_opt:.2e} (найкращий компроміс округлення і методу)")
    print(f"   Досягнута точність R0 = |y'_0(h0) - M'(t0)|: {r_opt:.2e}")

    # 3. Робочий крок h = 10^-3
    h_base = 1e-3
    print(f"\n3. Прийнято робочий крок h = {h_base:.0e} (0.001)")

    # 4. Обчислення з двома кроками: h та 2h
    y_h = diff_central(M, t0, h_base)
    y_2h = diff_central(M, t0, 2.0 * h_base)
    y_4h = diff_central(M, t0, 4.0 * h_base)

    # 5. Похибка при кроці h
    r1 = abs(y_h - exact_val)
    r_2h = abs(y_2h - exact_val)
    r_4h = abs(y_4h - exact_val)

    print("\n4-5. Центральна різниця для кроків h, 2h, 4h:")
    print(f"   y'_0(4h) = {y_4h:.12f}  |  Похибка R(4h) = {r_4h:.6e}")
    print(f"   y'_0(2h) = {y_2h:.12f}  |  Похибка R(2h) = {r_2h:.6e}")
    print(f"   y'_0(h)  = {y_h:.12f}  |  Похибка R1   = {r1:.6e}")
    print(f"   Співвідношення R(2h)/R(h)   = {r_2h / r1:.4f} (теоретично 2^2 = 4)")
    print(f"   Співвідношення R(4h)/R(2h)  = {r_4h / r_2h:.4f} (теоретично 2^2 = 4)")

    # 6. Метод Рунге-Ромберга
    y_rr = runge_romberg(y_h, y_2h, p=2)
    r2 = abs(y_rr - exact_val)
    print("\n6. Уточнення за методом Рунге-Ромберга (кроки h та 2h):")
    print(f"   y'_R = y'_0(h) + (y'_0(h) - y'_0(2h)) / 3 = {y_rr:.12f}")
    print(f"   Похибка R2 = |y'_R - M'(t0)| = {r2:.6e}")
    print(f"   Зменшення похибки порівняно з y'_0(h): у {r1 / r2:.1f} разів!")

    # 7. Метод Ейткена та оцінка порядку точності
    y_aitken, p_est = aitken(y_h, y_2h, y_4h)
    r3 = abs(y_aitken - exact_val)
    print("\n7. Метод Ейткена (кроки h, 2h, 4h):")
    print(f"   y'_E = {y_aitken:.12f}")
    print(f"   Оцінка порядку точності p: {p_est:.6f} (теоретичний порядок p = 2)")
    print(f"   Похибка R3 = |y'_E - M'(t0)| = {r3:.6e}")
    print(f"   Зменшення похибки порівняно з y'_0(h): у {r1 / r3:.1f} разів!")

    # 8. Додатковий розрахунок для кроків із демонстраційного прикладу (h = 0.01)
    h_demo = 0.01
    d_demo_h = diff_central(M, t0, h_demo)
    d_demo_h2 = diff_central(M, t0, h_demo / 2.0)
    d_demo_h4 = diff_central(M, t0, h_demo / 4.0)
    err_demo_h = abs(d_demo_h - exact_val)
    err_demo_h2 = abs(d_demo_h2 - exact_val)
    err_demo_h4 = abs(d_demo_h4 - exact_val)
    y_demo_rr = runge_romberg(d_demo_h2, d_demo_h, p=2)
    err_demo_rr = abs(y_demo_rr - exact_val)
    y_demo_aitken, p_demo = aitken(d_demo_h4, d_demo_h2, d_demo_h)
    err_demo_aitken = abs(y_demo_aitken - exact_val)

    # Побудова графіків
    plot_results(h_grid, err_grid, h_opt, r_opt, h_base, r1, r_2h, r_4h, r2, r3, t0, exact_val)
    print("\n[+] Графіки успішно збережено в папку plots/:")
    print("    - plots/regularization_curve.png")
    print("    - plots/methods_comparison.png")
    print("    - plots/moisture_dynamics.png")

    # Збереження результатів у файл results.txt
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТИ РОЗРАХУНКІВ ЛАБОРАТОРНОЇ РОБОТИ №4\n")
        f.write("Тема: Точність формул чисельного диференціювання. Рунге-Ромберг. Ейткен\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"1. Аналітичне значення M'(1): {exact_val:.12f}\n\n")
        f.write(f"2. Дослідження регуляризації кроку h in [10^-20 ... 10^3]:\n")
        f.write(f"   - Оптимальний крок h0: {h_opt:.2e}\n")
        f.write(f"   - Мінімальна похибка R0: {r_opt:.2e}\n\n")
        f.write(f"3. Розрахунки для h = 10^-3 (0.001):\n")
        f.write(f"   - y'_0(4h) = {y_4h:.12f},  похибка = {r_4h:.6e}\n")
        f.write(f"   - y'_0(2h) = {y_2h:.12f},  похибка = {r_2h:.6e}\n")
        f.write(f"   - y'_0(h)  = {y_h:.12f},  похибка R1 = {r1:.6e}\n")
        f.write(f"   - Рунге-Ромберг y'_R = {y_rr:.12f},  похибка R2 = {r2:.6e}\n")
        f.write(f"   - Ейткен y'_E        = {y_aitken:.12f},  похибка R3 = {r3:.6e}\n")
        f.write(f"   - Оцінка порядку p   = {p_est:.6f}\n\n")
        f.write("4. Порівняння результатів для демонстраційного прикладу (h = 0.01):\n")
        f.write(f"   - ЦР h=0.01:    {d_demo_h:.12f},  похибка = {err_demo_h:.6e}\n")
        f.write(f"   - ЦР h=0.005:   {d_demo_h2:.12f},  похибка = {err_demo_h2:.6e}\n")
        f.write(f"   - ЦР h=0.0025:  {d_demo_h4:.12f},  похибка = {err_demo_h4:.6e}\n")
        f.write(f"   - Рунге-Ромберг: {y_demo_rr:.12f},  похибка = {err_demo_rr:.6e}\n")
        f.write(f"   - Ейткен:        {y_demo_aitken:.12f},  похибка = {err_demo_aitken:.6e}\n")

    print("[+] Підсумкові числові дані збережено у 'results.txt'.")
    print("\nРоботу програми успішно завершено.")


if __name__ == "__main__":
    main()

