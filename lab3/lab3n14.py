import numpy as np
from scipy.optimize import minimize_scalar
import math

def f(x):
    return math.cos(5 * x**2)  

def f_8th_derivative(x): 
    h = 1e-5
    return (f(x - 4*h) - 8*f(x - 3*h) + 28*f(x - 2*h) - 56*f(x - h) + 70*f(x)
            - 56*f(x + h) + 28*f(x + 2*h) - 8*f(x + 3*h) + f(x + 4*h)) / (h**8)

a = 0
b = 5  
n = 4  

t = np.array([0.86113631, 0.33998104, -0.33998104, -0.86113631])
w = np.array([0.34785484, 0.65214516, 0.65214516, 0.34785484])

# Преобразование узлов
x_gauss = 0.5 * (b - a) * t + 0.5 * (b + a)

# Вычисляем интеграл
fx = [f(xi) for xi in x_gauss]
I_gauss = 0.5 * (b - a) * np.sum(w * fx)

# Находим максимум 8-й производной 
result = minimize_scalar(lambda x: -abs(f_8th_derivative(x)), bounds=(a, b), method='bounded')
max_f_8th = -result.fun

# Остаточный член для метода Гаусса 
R = 2.88e-7 * max_f_8th * ((b - a) / 2)**8  

print(f"Количество узлов: {n}")
print(f"Приближенное значение интеграла: {I_gauss:.6f}")
print(f"Остаточный член: {R:.6e}")