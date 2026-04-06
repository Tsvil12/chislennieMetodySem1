import numpy as np
from scipy.optimize import minimize_scalar
import math

def f(x):
    return math.cos(5 * x**2) 

def f_2nd_derivative(x):
    # Вторая производная для cos(5x²)
    # f(x) = cos(5x²)
    # f'(x) = -10x * sin(5x²)
    # f''(x) = -10*sin(5x²) - 100x²*cos(5x²)
    return -10 * math.sin(5 * x**2) - 100 * x**2 * math.cos(5 * x**2)

a = 0
b = 5  
epsilon = 1e-6
n = 1

# Находим максимум модуля второй производной на интервале [a, b]
result = minimize_scalar(lambda x: -abs(f_2nd_derivative(x)), bounds=(a, b), method='bounded')
max_f_2nd_derivative = -result.fun

while True:
    h = (b - a) / n

    # Оценка остаточного члена для метода левых прямоугольников
    R = -(b - a)**3 / (24 * n**2) * max_f_2nd_derivative

    # Проверка на погрешность
    if abs(R) > epsilon / 2:
        n += 1
        continue
    
    # Вычисляем значения функции в левой границе подынтервалов
    x = np.linspace(a, b - h, n)  # Левые границы
    fx = [f(xi) for xi in x]
    
    # Применяем метод прямоугольников 
    I_rect = sum(fx) * h
    break

print(f"Количество подотрезков: {n}")
print(f"Приближенное значение интеграла: {I_rect}")
print(f"Остаточный член: {R}")