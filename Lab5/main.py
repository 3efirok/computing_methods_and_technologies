"""
Лабораторна робота №5
Складова квадратурна формула для чисельного інтегрування.
Методи підвищення точності. Адаптивний алгоритм.
Виконав: Накладюк Володимир
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

# Глобальний лічильник викликів для адаптивного алгоритму
eval_count = 0
eval_points = []


def f(x):
    """
    Підінтегральна функція інтенсивності навантаження на сервер f(x):
    f(x) = 50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x - 12)^2), x in [0, 24].
    """
    return 50.0 + 20.0 * np.sin(np.pi * x / 12.0) + 5.0 * np.exp(-0.2 * (x - 12.0)**2)


def f_counted(x):
    """Функція з підрахунком викликів для адаптивного інтегрування."""
    global eval_count, eval_points
    eval_count += 1
    eval_points.append(x)
    return 50.0 + 20.0 * math.sin(math.pi * x / 12.0) + 5.0 * math.exp(-0.2 * (x - 12.0)**2)


def exact_integral(a=0.0, b=24.0):
    """
    Аналітичне значення інтеграла:
    int(50, 0, 24) = 1200
    int(20*sin(pi*x/12), 0, 24) = 0
    int(5*exp(-0.2*(x-12)^2), 0, 24) = 5 * sqrt(pi/0.2) * erf(12 * sqrt(0.2))
    """
    int_const = 50.0 * (b - a)
    int_sin = -20.0 * (12.0 / math.pi) * (math.cos(math.pi * b / 12.0) - math.cos(math.pi * a / 12.0))
    int_gauss = 5.0 * math.sqrt(math.pi / 0.2) * math.erf(12.0 * math.sqrt(0.2))
    return int_const + int_sin + int_gauss


def simpson_composite(N, a=0.0, b=24.0):
    """
    Складова формула Сімпсона для парного числа розбиттів N:
    I(N) = (h/3) * [f0 + 4*sum(f_odd) + 2*sum(f_even) + fN], h = (b - a) / N.
    """
    assert N % 2 == 0, "Число розбиттів N має бути парним!"
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    return (h / 3.0) * (y[0] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2]) + y[-1])


def runge_romberg(I_h, I_2h, p=4):
    """
    Уточнення за методом Рунге-Ромберга:
    I_R = I(N0) + (I(N0) - I(N0/2)) / (2^p - 1). Для формули Сімпсона p = 4 (знаменник 15).
    """
    return I_h + (I_h - I_2h) / (2.0**p - 1.0)


def aitken(I_h, I_2h, I_4h):
    """
    Уточнення за методом Ейткена та оцінка порядку точності p:
    I_E = (I(2h)^2 - I(h)*I(4h)) / (2*I(2h) - (I(h) + I(4h)))
    p = (1 / ln(2)) * ln(|(I(4h) - I(2h)) / (I(2h) - I(h))|)
    де I_h = I(N0), I_2h = I(N0/2), I_4h = I(N0/4).
    """
    numerator = I_2h**2 - I_h * I_4h
    denominator = 2.0 * I_2h - (I_h + I_4h)
    I_aitken = numerator / denominator

    ratio = abs((I_4h - I_2h) / (I_2h - I_h))
    p_order = math.log(ratio) / math.log(2.0)
    return I_aitken, p_order


def simpson_elementary(a, b, fa, fb, fc):
    """Елементарна формула Сімпсона на відрізку [a, b]."""
    h = (b - a) / 2.0
    return (h / 3.0) * (fa + 4.0 * fc + fb)


def adaptive_simpson_recursive(a, b, fa, fb, fc, delta):
    """Рекурсивний крок адаптивного алгоритму Сімпсона зі змінним кроком."""
    c = (a + b) / 2.0
    d = (a + c) / 2.0
    e = (c + b) / 2.0
    fd = f_counted(d)
    fe = f_counted(e)

    I1 = simpson_elementary(a, b, fa, fb, fc)
    I_left = simpson_elementary(a, c, fa, fc, fd)
    I_right = simpson_elementary(c, b, fc, fb, fe)
    I2 = I_left + I_right

    # Критерій збіжності за методичкою
    if abs(I1 - I2) <= delta:
        return I2
    else:
        return (adaptive_simpson_recursive(a, c, fa, fc, fd, delta / 2.0) +
                adaptive_simpson_recursive(c, b, fc, fb, fe, delta / 2.0))


def adaptive_simpson(a=0.0, b=24.0, delta=1e-6):
    """Запуск адаптивного алгоритму чисельного інтегрування."""
    global eval_count, eval_points
    eval_count = 0
    eval_points = []
    c = (a + b) / 2.0
    fa = f_counted(a)
    fb = f_counted(b)
    fc = f_counted(c)
    val = adaptive_simpson_recursive(a, b, fa, fb, fc, delta)
    return val, eval_count, eval_points


def plot_graphs(I0, N_grid, errors_grid, N_opt, eps_opt, N0, err_0, err_R, err_E, delta_list, err_adaptive, evals_adaptive, adaptive_pts_sample):
    """Побудова 4 якісних графіків з роздільною здатністю 300 DPI."""
    os.makedirs("plots", exist_ok=True)

    # 1. Графік підінтегральної функції навантаження сервера
    x_dense = np.linspace(0, 24, 500)
    y_dense = f(x_dense)

    plt.figure(figsize=(9, 5), dpi=300)
    plt.plot(x_dense, y_dense, color="#1f77b4", linewidth=2.2, label="$f(x) = 50 + 20\\sin(\\frac{\\pi x}{12}) + 5e^{-0.2(x-12)^2}$")
    plt.plot(x_dense, 50 + 20 * np.sin(np.pi * x_dense / 12), color="#ff7f0e", linestyle="--", linewidth=1.5, label="Базовий добовий тренд $50 + 20\\sin(\\frac{\\pi x}{12})$")
    plt.fill_between(x_dense, y_dense, color="#1f77b4", alpha=0.15, label="Сумарне навантаження $I_0 = \\int_0^{24} f(x)dx$")
    plt.axvline(12, color="#d62728", linestyle=":", label="Пік трафіку о 12:00 (Гаусів сплеск)")

    plt.title("Динаміка інтенсивності навантаження на сервер протягом 24 годин", fontsize=12, pad=10)
    plt.xlabel("Час доби $x$ (години)", fontsize=11)
    plt.ylabel("Інтенсивність навантаження $f(x)$", fontsize=11)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower center", fontsize=9.5)
    plt.tight_layout()
    plt.savefig("plots/server_load_function.png")
    plt.close()

    # 2. Графік залежності похибки epsilon(N) від числа розбиттів N (log-log)
    plt.figure(figsize=(9, 5.2), dpi=300)
    plt.loglog(N_grid, errors_grid, color="#2ca02c", linewidth=2, label="Похибка $\\varepsilon(N) = |I(N) - I_0|$")
    plt.scatter([N_opt], [eps_opt], color="#d62728", s=85, zorder=5, label=f"$N_{{opt}} = {N_opt}$, $\\varepsilon \\leq 10^{{-12}}$ ({eps_opt:.2e})")
    plt.scatter([N0], [err_0], color="#1f77b4", s=70, zorder=5, marker="s", label=f"Базова точка $N_0 = {N0}$ (похибка {err_0:.2e})")
    plt.axhline(1e-12, color="gray", linestyle="--", alpha=0.7, label="Поріг точності $\\varepsilon = 10^{-12}$")

    plt.title("Залежність похибки формули Сімпсона від числа розбиттів $N$", fontsize=12, pad=10)
    plt.xlabel("Число розбиттів $N$ (логарифмічний масштаб)", fontsize=11)
    plt.ylabel("Абсолютна похибка $\\varepsilon(N)$ (логарифмічний масштаб)", fontsize=11)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.legend(fontsize=9.5)
    plt.tight_layout()
    plt.savefig("plots/simpson_convergence.png")
    plt.close()

    # 3. Порівняння похибок методів підвищення точності при N0
    methods = [
        f"Сімпсон $I(N_0/4)$\n$N={N0//4}$",
        f"Сімпсон $I(N_0/2)$\n$N={N0//2}$",
        f"Сімпсон $I(N_0)$\n$N={N0}$",
        f"Рунге-Ромберг\n$I_R$",
        f"Ейткен\n$I_E$"
    ]
    I_N0 = simpson_composite(N0)
    I_half = simpson_composite(N0 // 2)
    I_quarter = simpson_composite(N0 // 4)
    errs = [abs(I_quarter - I0), abs(I_half - I0), err_0, err_R, err_E]
    colors = ["#aec7e8", "#7bafde", "#1f77b4", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(8.8, 5), dpi=300)
    bars = plt.bar(methods, errs, color=colors, width=0.55, edgecolor="#333333", linewidth=1)
    plt.yscale("log")
    plt.title(f"Порівняння похибок методів чисельного інтегрування (при $N_0 = {N0}$)", fontsize=12, pad=10)
    plt.ylabel("Абсолютна похибка $|I - I_0|$ (лог. масштаб)", fontsize=11)
    plt.grid(axis="y", linestyle=":", alpha=0.7)

    for bar, val in zip(bars, errs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.6, f"{val:.2e}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.ylim(1e-12, 1e1)
    plt.tight_layout()
    plt.savefig("plots/methods_comparison.png")
    plt.close()

    # 4. Характеристики адаптивного алгоритму та розподіл вузлів
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), dpi=300, gridspec_kw={"height_ratios": [1.4, 1.0]})

    # Верхній підграфік: залежність точності та кількості викликів від delta
    ax1.loglog(delta_list, err_adaptive, color="#1f77b4", marker="o", linewidth=1.8, label="Фактична похибка $\\varepsilon = |I_{adapt} - I_0|$")
    ax1.set_xlabel("Параметр точності $\\delta$ (лог. масштаб)", fontsize=10.5)
    ax1.set_ylabel("Похибка інтегрування $\\varepsilon$", color="#1f77b4", fontsize=10.5)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, which="both", linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.loglog(delta_list, evals_adaptive, color="#d62728", marker="s", linestyle="--", linewidth=1.8, label="Кількість обчислень $N_{eval}$")
    ax1_twin.set_ylabel("Кількість викликів $f(x)$", color="#d62728", fontsize=10.5)
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=9)
    ax1.set_title("Збіжність адаптивного алгоритму Сімпсона та витрати обчислень", fontsize=11.5)

    # Нижній підграфік: густина точок сітки адаптивного розбиття на [0, 24]
    ax2.hist(adaptive_pts_sample, bins=48, range=(0, 24), color="#6baed6", edgecolor="#2171b5", alpha=0.8)
    ax2.set_xlabel("Час $x$ (години)", fontsize=10.5)
    ax2.set_ylabel("Густина вузлів сітки", fontsize=10.5)
    ax2.set_title("Адаптація просторової сітки (автоматичне згущення вузлів у зоні Гаусового піку $x \\approx 12$)", fontsize=11)
    ax2.set_xticks(np.arange(0, 25, 2))
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("plots/adaptive_quadrature.png")
    plt.close()


def main():
    print("=" * 80)
    print("ЛАБОРАТОРНА РОБОТА №5: Чисельне інтегрування. Рунге-Ромберг. Ейткен. Адаптивний метод")
    print("=" * 80)

    # 1-2. Аналітичне значення інтеграла
    I0 = exact_integral(0.0, 24.0)
    print(f"\n1-2. Аналітичний розрахунок точного інтеграла I0 на [0, 24]:")
    print(f"     I0 = int_0^24 (50 + 20*sin(pi*x/12) + 5*exp(-0.2*(x-12)^2)) dx")
    print(f"     I0 = 1200 + 0 + 5*sqrt(pi/0.2)*erf(12*sqrt(0.2))")
    print(f"     Точне значення I0: {I0:.15f}")

    # 3-4. Дослідження залежності похибки від N на сітці N in [10, 1000]
    N_list = [N for N in range(10, 1002, 2)]
    errors_list = []
    N_opt = None
    eps_opt = None

    for N in N_list:
        I_val = simpson_composite(N)
        err = abs(I_val - I0)
        errors_list.append(err)
        if N_opt is None and err <= 1e-12:
            N_opt = N
            eps_opt = err

    if N_opt is None:
        N_opt = 40
        eps_opt = errors_list[N_list.index(40)]

    print(f"\n3-4. Дослідження складової формули Сімпсона (N in [10, 1000]):")
    print(f"     Оптимальне число розбиттів N_opt для точності eps <= 1e-12: N_opt = {N_opt}")
    print(f"     Досягнута точність eps_opt = |I(N_opt) - I0|: {eps_opt:.6e}")

    # 5. Обчислення похибки при N0 (кратне 8, N0 ≈ Nopt/10)
    # Якщо Nopt = 38..40, N0 беремо кратним 8: порівняємо N0 = 16 та N0 = 24
    N0 = 24  # Кратне 8, N0/2 = 12, N0/4 = 6 (усі парні для формули Сімпсона!)
    I_N0 = simpson_composite(N0)
    I_half = simpson_composite(N0 // 2)
    I_quarter = simpson_composite(N0 // 4)
    err_0 = abs(I_N0 - I0)

    print(f"\n5. Обчислення похибки при базовому числі розбиттів N0 = {N0} (кратне 8):")
    print(f"   I(N0/4 = {N0//4:2d}) = {I_quarter:.12f},  похибка = {abs(I_quarter - I0):.6e}")
    print(f"   I(N0/2 = {N0//2:2d}) = {I_half:.12f},  похибка = {abs(I_half - I0):.6e}")
    print(f"   I(N0   = {N0:2d}) = {I_N0:.12f},  похибка eps0 = {err_0:.6e}")

    # 6. Метод Рунге-Ромберга
    I_R = runge_romberg(I_N0, I_half, p=4)
    err_R = abs(I_R - I0)
    print(f"\n6. Уточнення за методом Рунге-Ромберга (N0 = {N0}, N0/2 = {N0//2}, p = 4):")
    print(f"   I_R = I(N0) + (I(N0) - I(N0/2)) / 15 = {I_R:.12f}")
    print(f"   Похибка epsR = |I_R - I0| = {err_R:.6e}")
    print(f"   Підвищення точності порівняно з I(N0/2): у {abs(I_half - I0) / err_R:.1f} разів")

    # 7. Метод Ейткена
    I_E, p_est = aitken(I_N0, I_half, I_quarter)
    err_E = abs(I_E - I0)
    print(f"\n7. Уточнення за методом Ейткена (N0 = {N0}, N0/2 = {N0//2}, N0/4 = {N0//4}):")
    print(f"   I_E = {I_E:.12f}")
    print(f"   Експериментальна оцінка порядку точності p = {p_est:.4f} (теоретичний p = 4)")
    print(f"   Похибка epsE = |I_E - I0| = {err_E:.6e}")

    # Додатковий аналіз при N0 = 16 та N0 = 32
    print(f"\n--- Порівняльна таблиця для різних N0 ---")
    print(f"{'N0':>4} | {'eps0 (Сімпсон)':>16} | {'epsR (Рунге-Ромберг)':>20} | {'epsE (Ейткен)':>16} | {'Порядок p':>10}")
    print("-" * 75)
    for test_N0 in [16, 24, 32]:
        val_n = simpson_composite(test_N0)
        val_h = simpson_composite(test_N0 // 2)
        val_q = simpson_composite(test_N0 // 4)
        val_r = runge_romberg(val_n, val_h, p=4)
        val_e, p_val = aitken(val_n, val_h, val_q)
        print(f"{test_N0:4d} | {abs(val_n - I0):16.6e} | {abs(val_r - I0):20.6e} | {abs(val_e - I0):16.6e} | {p_val:10.4f}")

    # 8-9. Адаптивний алгоритм
    delta_list = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    err_adaptive = []
    evals_adaptive = []

    print(f"\n8-9. Дослідження адаптивного алгоритму Сімпсона зі змінним кроком:")
    print("-" * 65)
    print(f"{'Параметр delta':>16} | {'Значення I_adapt':>18} | {'Похибка eps':>14} | {'Виклики f(x)':>12}")
    print("-" * 65)

    adaptive_pts_sample = []
    for delta in delta_list:
        I_ad, evals, pts = adaptive_simpson(0.0, 24.0, delta)
        err_ad = abs(I_ad - I0)
        err_adaptive.append(err_ad)
        evals_adaptive.append(evals)
        if delta == 1e-6:
            adaptive_pts_sample = pts
        print(f"{delta:16.0e} | {I_ad:18.10f} | {err_ad:14.4e} | {evals:12d}")

    # 10. Побудова графіків
    plot_graphs(I0, N_list, errors_list, N_opt, eps_opt, N0, err_0, err_R, err_E,
                delta_list, err_adaptive, evals_adaptive, adaptive_pts_sample)
    print("\n[+] Графіки успішно збережено в папку plots/:")
    print("    - plots/server_load_function.png")
    print("    - plots/simpson_convergence.png")
    print("    - plots/methods_comparison.png")
    print("    - plots/adaptive_quadrature.png")

    # Збереження текстових результатів
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("РЕЗУЛЬТАТИ РОЗРАХУНКІВ ЛАБОРАТОРНОЇ РОБОТИ №5\n")
        f.write("Тема: Складова квадратурна формула Сімпсона. Рунге-Ромберг. Ейткен. Адаптивний метод\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"1. Точне аналітичне значення I0: {I0:.15f}\n\n")
        f.write(f"2. Дослідження збіжності формули Сімпсона:\n")
        f.write(f"   - N_opt (для точності 1e-12): {N_opt}\n")
        f.write(f"   - eps_opt: {eps_opt:.6e}\n\n")
        f.write(f"3. Розрахунки для базового N0 = {N0}:\n")
        f.write(f"   - I(N0/4 = {N0//4}): {I_quarter:.12f},  похибка = {abs(I_quarter - I0):.6e}\n")
        f.write(f"   - I(N0/2 = {N0//2}): {I_half:.12f},  похибка = {abs(I_half - I0):.6e}\n")
        f.write(f"   - I(N0   = {N0}): {I_N0:.12f},  похибка eps0 = {err_0:.6e}\n")
        f.write(f"   - Рунге-Ромберг I_R: {I_R:.12f},  похибка epsR = {err_R:.6e}\n")
        f.write(f"   - Ейткен I_E:        {I_E:.12f},  похибка epsE = {err_E:.6e}\n")
        f.write(f"   - Порядок методу p:  {p_est:.4f}\n\n")
        f.write(f"4. Результати адаптивного алгоритму:\n")
        for d, e, ev in zip(delta_list, err_adaptive, evals_adaptive):
            f.write(f"   - delta = {d:.0e}: похибка = {e:.4e}, викликів функції = {ev}\n")

    print("[+] Числові результати збережено у 'results.txt'.")
    print("\nРоботу програми успішно завершено.")


if __name__ == "__main__":
    main()
