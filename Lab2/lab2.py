"""
Лабораторна робота №2
Метод інтерполяції: многочлени Ньютона та факторіальні многочлени
Варіант 2: Планування обчислювальних ресурсів (DevOps).
           Прогнозування навантаження на CPU
Виконав: Накладюк Володимир
"""

import csv
import math
import os
import matplotlib.pyplot as plt
import numpy as np


# ==================== Зчитування даних ====================
def read_data(filename):
    """Зчитування даних з CSV-файлу"""
    x = []
    y = []
    with open(filename, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['RPS']))
            y.append(float(row['CPU']))
    return x, y


# ==================== Розділені різниці ====================
def divided_differences(x, y):
    """Побудова таблиці розділених різниць"""
    n = len(x)
    table = [[0.0] * n for _ in range(n)]

    for i in range(n):
        table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i + 1][j - 1] - table[i][j - 1]) / (x[i + j] - x[i])

    return table


def print_divided_differences(x, y, table):
    """Виведення таблиці розділених різниць"""
    n = len(x)
    print("\nТаблиця розділених різниць:")
    print("-" * 80)

    header = f"{'x':>8} | {'f(x)':>10}"
    for j in range(1, n):
        header += f" | {'Δ' + str(j):>10}"
    print(header)
    print("-" * 80)

    for i in range(n):
        row = f"{x[i]:>8.1f} | {table[i][0]:>10.4f}"
        for j in range(1, n - i):
            row += f" | {table[i][j]:>10.6f}"
        print(row)
    print("-" * 80)


# ==================== Многочлен Ньютона ====================
def newton_interpolation(x_nodes, table, x_val):
    """
    Інтерполяційний многочлен Ньютона (інтерполяція вперед).
    P(x) = f(x0) + (x-x0)*f(x0,x1) + (x-x0)(x-x1)*f(x0,x1,x2) + ...
    """
    n = len(x_nodes)
    result = table[0][0]
    product = 1.0

    for j in range(1, n):
        product *= (x_val - x_nodes[j - 1])
        result += table[0][j] * product

    return result


# ==================== Факторіальні многочлени ====================
def finite_differences(y):
    """Обчислення скінченних різниць"""
    n = len(y)
    diff_table = [[0.0] * n for _ in range(n)]

    for i in range(n):
        diff_table[i][0] = y[i]

    for j in range(1, n):
        for i in range(n - j):
            diff_table[i][j] = diff_table[i + 1][j - 1] - diff_table[i][j - 1]

    return diff_table


def factorial_polynomial_interpolation(x_nodes, y_nodes, x_val):
    """
    Інтерполяція факторіальними многочленами для рівновіддалених вузлів.
    P(x) = f0 + t*Δf0/1! + t(t-1)*Δ²f0/2! + ...
    де t = (x - x0) / h
    """
    n = len(x_nodes)
    h = x_nodes[1] - x_nodes[0]

    diff_table = finite_differences(y_nodes)

    t = (x_val - x_nodes[0]) / h

    result = diff_table[0][0]
    factorial_product = 1.0

    for k in range(1, n):
        factorial_product *= (t - (k - 1))
        result += diff_table[0][k] * factorial_product / math.factorial(k)

    return result


def print_finite_differences(y_nodes, diff_table):
    """Виведення таблиці скінченних різниць"""
    n = len(y_nodes)
    print("\nТаблиця скінченних різниць:")
    print("-" * 60)

    header = f"{'y':>10}"
    for j in range(1, n):
        header += f" | {'Δ^' + str(j) + 'y':>10}"
    print(header)
    print("-" * 60)

    for i in range(n):
        row = f"{diff_table[i][0]:>10.4f}"
        for j in range(1, n - i):
            row += f" | {diff_table[i][j]:>10.4f}"
        print(row)
    print("-" * 60)


# ==================== Тестова функція для дослідження Рунге ====================
def test_function(x):
    """
    Тестова функція для дослідження ефекту Рунге:
    f(x) = 1 / (1 + 0.0001*x^2)
    """
    return 1.0 / (1.0 + 0.0001 * x * x)


# ==================== Основна програма ====================
def main():
    os.makedirs('images', exist_ok=True)

    # ===== ЧАСТИНА 1: Основне завдання =====
    print("=" * 60)
    print("ЧАСТИНА 1: ОСНОВНЕ ЗАВДАННЯ")
    print("=" * 60)

    # 1. Зчитування даних
    x, y = read_data("data.csv")
    print("Вхідні дані:")
    print(f"  RPS:  {x}")
    print(f"  CPU%: {y}")
    n = len(x)

    # 2. Побудова таблиці розділених різниць
    table = divided_differences(x, y)
    print_divided_differences(x, y, table)

    # 3. Прогноз для 600 RPS — метод Ньютона
    x_target = 600.0
    cpu_newton = newton_interpolation(x, table, x_target)
    print(f"\n--- Прогноз для RPS = {x_target:.0f} ---")
    print(f"Многочлен Ньютона:          CPU = {cpu_newton:.4f} %")

    # 4. Факторіальні многочлени
    # Створюємо рівновіддалені вузли для факторіальних многочленів
    x_eq = [50.0 + i * (800.0 - 50.0) / (n - 1) for i in range(n)]
    y_eq = [newton_interpolation(x, table, xi) for xi in x_eq]

    diff_table = finite_differences(y_eq)
    print_finite_differences(y_eq, diff_table)

    cpu_factorial = factorial_polynomial_interpolation(x_eq, y_eq, x_target)
    print(f"Факторіальні многочлени:    CPU = {cpu_factorial:.4f} %")

    diff_methods = abs(cpu_newton - cpu_factorial)
    print(f"Різниця між методами:       |ΔP| = {diff_methods:.6f}")

    # Графік 1: CPU(RPS) з інтерполяційною кривою
    x_plot = np.linspace(min(x), max(x), 300)
    y_newton_plot = [newton_interpolation(x, table, xi) for xi in x_plot]

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, y_newton_plot, 'b-', linewidth=2, label='Многочлен Ньютона')
    plt.plot(x, y, 'ro', markersize=8, label='Експериментальні точки')
    plt.plot(x_target, cpu_newton, 'g^', markersize=12,
             label=f'Прогноз: CPU({x_target:.0f}) = {cpu_newton:.2f}%')
    plt.xlabel('RPS (запити/с)', fontsize=12)
    plt.ylabel('CPU (%)', fontsize=12)
    plt.title('Інтерполяція CPU = f(RPS) — Варіант 2 (Накладюк Володимир)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/graph_interpolation.png', dpi=300)
    plt.close()
    print("\nГрафік інтерполяції збережено: images/graph_interpolation.png")

    # ===== ЧАСТИНА 2: ДОСЛІДНИЦЬКА ЧАСТИНА =====
    print("\n" + "=" * 60)
    print("ЧАСТИНА 2: ДОСЛІДНИЦЬКА ЧАСТИНА")
    print("=" * 60)

    a, b = 50.0, 800.0
    x_fine = np.linspace(a, b, 500)
    y_exact = [test_function(xi) for xi in x_fine]

    nodes_counts = [5, 10, 20]

    # Дослідження 1: Вплив кількості вузлів (фіксований інтервал)
    print("\n--- Дослідження 1: Вплив кількості вузлів ---")
    print(f"Інтервал: [{a}, {b}]")
    print(f"Тестова функція: f(x) = 1 / (1 + 0.0001*x^2)")

    max_errors = {}

    for num in nodes_counts:
        h = (b - a) / (num - 1)
        x_nodes = [a + i * h for i in range(num)]
        y_nodes = [test_function(xi) for xi in x_nodes]

        table_n = divided_differences(x_nodes, y_nodes)

        errors = [abs(test_function(xi) - newton_interpolation(x_nodes, table_n, xi))
                  for xi in x_fine]
        max_err = max(errors)
        max_errors[num] = max_err

        print(f"\n  n = {num}, h = {h:.2f}:")
        print(f"    Макс. похибка = {max_err:.8f}")

    # Графік 2: Порівняння інтерполяції при різній кількості вузлів
    plt.figure(figsize=(12, 7))
    colors = ['blue', 'green', 'orange']

    plt.plot(x_fine, y_exact, 'k-', linewidth=2, label='Точна f(x)')

    for idx, num in enumerate(nodes_counts):
        h = (b - a) / (num - 1)
        x_nodes = [a + i * h for i in range(num)]
        y_nodes = [test_function(xi) for xi in x_nodes]
        table_n = divided_differences(x_nodes, y_nodes)

        y_interp = [newton_interpolation(x_nodes, table_n, xi) for xi in x_fine]
        plt.plot(x_fine, y_interp, color=colors[idx], linewidth=1.5,
                 label=f'n = {num} (h={h:.1f})')
        plt.plot(x_nodes, y_nodes, 'o', color=colors[idx], markersize=4)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x)', fontsize=12)
    plt.title('Порівняння інтерполяції при різній кількості вузлів', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/graph_nodes_comparison.png', dpi=300)
    plt.close()
    print("\nГрафік порівняння збережено: images/graph_nodes_comparison.png")

    # Графік 3: Похибки
    plt.figure(figsize=(12, 7))

    for idx, num in enumerate(nodes_counts):
        h = (b - a) / (num - 1)
        x_nodes = [a + i * h for i in range(num)]
        y_nodes = [test_function(xi) for xi in x_nodes]
        table_n = divided_differences(x_nodes, y_nodes)

        errors = [abs(test_function(xi) - newton_interpolation(x_nodes, table_n, xi))
                  for xi in x_fine]
        plt.plot(x_fine, errors, color=colors[idx], linewidth=1.5,
                 label=f'|ε(x)| при n = {num}')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('|Похибка|', fontsize=12)
    plt.title('Графік похибок при різній кількості вузлів (ефект Рунге)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('images/graph_errors.png', dpi=300)
    plt.close()
    print("Графік похибок збережено: images/graph_errors.png")

    # Дослідження 2: Вплив кроку
    print("\n--- Дослідження 2: Вплив кроку ---")
    print(f"{'n':>5} | {'Крок h':>10} | {'Макс. похибка':>15}")
    print("-" * 40)

    for num in [5, 8, 10, 15, 20]:
        h = (b - a) / (num - 1)
        x_nodes = [a + i * h for i in range(num)]
        y_nodes = [test_function(xi) for xi in x_nodes]
        table_n = divided_differences(x_nodes, y_nodes)

        errors = [abs(test_function(xi) - newton_interpolation(x_nodes, table_n, xi))
                  for xi in x_fine]
        max_err = max(errors)
        print(f"{num:>5} | {h:>10.2f} | {max_err:>15.8f}")

    # Дослідження 3: Вплив інтервалу
    print("\n--- Дослідження 3: Вплив інтервалу ---")
    print(f"{'Інтервал':>20} | {'n':>5} | {'Макс. похибка':>15}")
    print("-" * 50)

    for k in [0, 1, 2]:
        a_new = a - k * 100
        b_new = b + k * 100
        h_fixed = 75.0
        num_new = int((b_new - a_new) / h_fixed) + 1
        x_nodes = [a_new + i * h_fixed for i in range(num_new)]
        y_nodes = [test_function(xi) for xi in x_nodes]

        table_n = divided_differences(x_nodes, y_nodes)
        x_test = np.linspace(a, b, 300)
        errors = [abs(test_function(xi) - newton_interpolation(x_nodes, table_n, xi))
                  for xi in x_test]
        max_err = max(errors)
        interval_str = f"[{a_new:.0f}, {b_new:.0f}]"
        print(f"{interval_str:>20} | {num_new:>5} | {max_err:>15.8f}")

    # Графік 4: Ефект Рунге
    plt.figure(figsize=(12, 7))

    x_ext = np.linspace(a - 50, b + 50, 500)
    y_exact_ext = [test_function(xi) for xi in x_ext]
    plt.plot(x_ext, y_exact_ext, 'k-', linewidth=2, label='Точна f(x)')

    for idx, num in enumerate([5, 10, 20]):
        h = (b - a) / (num - 1)
        x_nodes = [a + i * h for i in range(num)]
        y_nodes = [test_function(xi) for xi in x_nodes]
        table_n = divided_differences(x_nodes, y_nodes)

        y_interp = [newton_interpolation(x_nodes, table_n, xi) for xi in x_ext]
        plt.plot(x_ext, y_interp, color=colors[idx], linewidth=1.5,
                 label=f'n = {num}')

    plt.xlabel('x', fontsize=12)
    plt.ylabel('f(x) / Pn(x)', fontsize=12)
    plt.title('Ефект Рунге: коливання полінома при збільшенні кількості вузлів', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, 2.0)
    plt.tight_layout()
    plt.savefig('images/graph_runge_effect.png', dpi=300)
    plt.close()
    print("\nГрафік ефекту Рунге збережено: images/graph_runge_effect.png")

    # Прогноз для вихідних даних при різній кількості вузлів
    print("\n--- Прогноз CPU(600) при різній кількості вузлів ---")
    print("Використання підмножин вихідних даних:")

    for k in [3, 4, 5]:
        x_sub = x[:k]
        y_sub = y[:k]
        table_sub = divided_differences(x_sub, y_sub)
        pred = newton_interpolation(x_sub, table_sub, x_target)
        print(f"  n = {k} вузлів ({x_sub}): CPU({x_target:.0f}) = {pred:.4f}%")

    # ===== ВИСНОВКИ =====
    print("\n" + "=" * 60)
    print("ВИСНОВКИ")
    print("=" * 60)
    print(f"1. Побудовано модель CPU = f(RPS) за допомогою інтерполяції.")
    print(f"2. Прогноз CPU при 600 RPS:")
    print(f"   - Многочлен Ньютона:       {cpu_newton:.4f}%")
    print(f"   - Факторіальні многочлени: {cpu_factorial:.4f}%")
    print(f"   - Різниця між методами:    {diff_methods:.6f}")
    print(f"3. На тестовій функції досліджено ефект Рунге:")
    print(f"   - При n=5  макс. похибка = {max_errors[5]:.8f}")
    print(f"   - При n=10 макс. похибка = {max_errors[10]:.8f}")
    print(f"   - При n=20 макс. похибка = {max_errors[20]:.8f}")
    print(f"4. При збільшенні степеня полінома похибка на краях")
    print(f"   інтервалу зростає — це характерний ефект Рунге.")
    print(f"5. Модель є стабільною в межах інтервалу [50, 800] RPS")
    print(f"   і дає адекватний прогноз навантаження CPU.")


if __name__ == "__main__":
    main()

