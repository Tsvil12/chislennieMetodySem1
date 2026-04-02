import numpy as np
from scipy.optimize import minimize_scalar
import math

def f(x):
    return math.cos(5 * x**2)  # Ваша функция

def f_4th_derivative(x):
    # Для cos(5x²) четвертая производная сложна, используем численное дифференцирование
    h = 1e-5
    # Формула для четвертой производной через конечные разности
    return (f(x - 2*h) - 4*f(x - h) + 6*f(x) - 4*f(x + h) + f(x + 2*h)) / (h**4)

a = 0
b = 5  # Ваш интервал
epsilon = 1e-6
n = 2  # Начинаем с 2 (метод Симпсона требует четное n)

# Находим максимум модуля четвертой производной на интервале [a, b]
result = minimize_scalar(lambda x: -abs(f_4th_derivative(x)), bounds=(a, b), method='bounded')
max_f_4th_derivative = -result.fun

while True:
    h = (b - a) / n
    
    # Вычисляем значения функции в нужных точках
    x = np.linspace(a, b, n + 1)
    fx = [f(xi) for xi in x]

    R = -((b - a) / 180) * max_f_4th_derivative * h**4

    # Проверка на погрешность
    if abs(R) > epsilon / 2:
        n *= 2
        continue
    
    # Применяем формулу Симпсона (составную)
    I_simpson = (h / 3) * (fx[0] + 4 * sum(fx[1:-1:2]) + 2 * sum(fx[2:-2:2]) + fx[-1])
    break

print(f"Количество подотрезков: {n}")
print(f"Приближенное значение интеграла: {I_simpson:.6f}")
print(f"Остаточный член: {R}")