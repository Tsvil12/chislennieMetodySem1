import numpy as np
import matplotlib.pyplot as plt
import math

def f(x):
    return np.cos(np.sin(x)**2)

# Лагранж
def lagrange(x, x_nodes, y_nodes):
    n = len(x_nodes)
    result = 0
    for i in range(n):
        term = y_nodes[i]
        for j in range(n):
            if i != j:
                term *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += term
    return result

def find_max_M(n):
    def f_nth_derivative(x, n):
        h = 1e-5
        if n == 0:
            return f(x)
        elif n == 1:
            return (f(x + h) - f(x - h)) / (2 * h)
        else:
            return (f_nth_derivative(x + h, n - 1) - f_nth_derivative(x - h, n - 1)) / (2 * h)
    
    samples = 1000
    x_samples = np.linspace(0, math.pi, samples)
    max_val = 0
    
    for x in x_samples:
        try:
            deriv_val = abs(f_nth_derivative(x, n))
            if deriv_val > max_val:
                max_val = deriv_val
        except:
            continue
    
    return max_val

a, b = 0, math.pi  
n = 10 

# Узлы интерполяции
x_nodes = np.linspace(a, b, n)
y_nodes = f(x_nodes)

x_plot = np.linspace(a, b, 200)
y_true = f(x_plot)

# Интерполяция во всех точках
y_lagr = []
for x in x_plot:
    y_lagr.append(lagrange(x, x_nodes, y_nodes))

errors = np.abs(y_true - y_lagr)
max_error = np.max(errors)

M = find_max_M(n)
print(f"Найденное M для n={n}: {M:.2f}")

# Теоретическая погрешность
def theoretical_error(x):
    omega = 1.0
    for xi in x_nodes:
        omega *= (x - xi)
    
    return abs(omega) * M / math.factorial(n)

y_theory = [theoretical_error(x) for x in x_plot]
max_theory = np.max(y_theory)

# Вывод результатов
print("="*50)
print("РЕЗУЛЬТАТЫ ИНТЕРПОЛЯЦИИ")
print("="*50)
print(f"Функция: f(x) = cos(sin²x)")
print(f"Отрезок: [0, π]")
print(f"Количество узлов: {n}")
print(f"Найденное M = {M:.2f}")
print(f"Макс. реальная ошибка: {max_error:.2e}")
print(f"Макс. теоретическая оценка: {max_theory:.2e}")
print("="*50)

# Графики
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Функция и интерполяция
ax1.plot(x_plot, y_true, 'b-', linewidth=2, label='Точная функция')
ax1.plot(x_plot, y_lagr, 'r--', linewidth=2, label=f'Интерполяция L_{n}(x)')
ax1.plot(x_nodes, y_nodes, 'ko', markersize=6, label='Узлы интерполяции')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Функция и интерполяционный многочлен')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Ошибка интерполяции
ax2.plot(x_plot, errors, 'g-', linewidth=2)
ax2.fill_between(x_plot, 0, errors, alpha=0.3, color='green')
ax2.set_xlabel('x')
ax2.set_ylabel('Ошибка')
ax2.set_title(f'Реальная ошибка')
ax2.grid(True, alpha=0.3)

# 3. Сравнение в узлах
ax3.plot(x_nodes, y_nodes, 'bo', markersize=8, label='Точные значения')
ax3.plot(x_nodes, [lagrange(x, x_nodes, y_nodes) for x in x_nodes], 
         'rx', markersize=8, linewidth=2, label='Интерполяция в узлах')
for i in range(len(x_nodes)):
    ax3.plot([x_nodes[i], x_nodes[i]], 
             [y_nodes[i], lagrange(x_nodes[i], x_nodes, y_nodes)], 
             'k--', alpha=0.5)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Точки интерполяции')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Теоретическая оценка ошибки
ax4.plot(x_plot, errors, 'g-', linewidth=2, label='Реальная ошибка')
ax4.plot(x_plot, y_theory, 'm--', linewidth=2, label='Теоретическая оценка')
ax4.set_xlabel('x')
ax4.set_ylabel('Ошибка')
ax4.set_title('Сравнение с теорией')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Таблица значений
print("\nТАБЛИЦА ЗНАЧЕНИЙ В УЗЛАХ:")
print("-"*40)
print("№ узла\t   x\t\t   f(x)")
print("-"*40)
for i in range(len(x_nodes)):
    print(f"{i:3}\t{x_nodes[i]:8.4f}\t{y_nodes[i]:12.6f}")
print("-"*40)