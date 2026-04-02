import numpy as np
from scipy.optimize import minimize_scalar
import math

def f(x):
    return math.cos(5 * x**2)  

def f_double_prime(x):
   
    return -10 * math.sin(5 * x**2) - 100 * x**2 * math.cos(5 * x**2)

a = 0
b = 5  
epsilon = 1e-6

# Находим максимум второй производной на интервале [a, b]
result = minimize_scalar(lambda x: -abs(f_double_prime(x)), bounds=(a, b), method='bounded')
max_f_double_prime = -result.fun

# Вычисляем шаг h на основе заданной точности
n = 1
h = (b - a) / n
R = (b - a) / 12 * max_f_double_prime * h**2

# Увеличиваем n, пока остаточный член не станет меньше epsilon/2
while R > epsilon / 2:
    n += 1
    h = (b - a) / n
    R = (b - a) / 12 * max_f_double_prime * h**2

# Вычисляем значения функции в узлах
x_values = [a + i * h for i in range(n + 1)]
f_values = [f(x) for x in x_values]

# Применяем формулу трапеции
integral_approximation = (h / 2) * (f_values[0] + f_values[-1] + 2 * sum(f_values[1:-1]))

# Выводим результаты
print(f"Количество подотрезков: {n}")
print(f"Приближённое значение интеграла: {integral_approximation:.6f}")
print(f"Остаточный член: {R}")