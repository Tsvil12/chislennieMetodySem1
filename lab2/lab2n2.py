import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(np.sin(x)**2)

a = 0
b = np.pi
n = 2
x0 = 1.5  
epsilon = 1e-10

print(f"Функция: y = cos(sin²x)")
print(f"Отрезок: [{a}, {b}]")
print(f"Точка x0 = {x0}")
print(f"Точность ε = {epsilon}")
print(f"n = {n}")
print()

#корни полинома Лежандра степени 2
t1 = -1/np.sqrt(3)
t2 = 1/np.sqrt(3)

# Переводим из [-1,1] в [a,b]
x1 = (b - a)/2 * t1 + (a + b)/2
x2 = (b - a)/2 * t2 + (a + b)/2

print("Узлы Гаусса (на [a,b]):")
print(f"x1 = {x1:.6f}")
print(f"x2 = {x2:.6f}")

# Значения функции в узлах
y1 = f(x1)
y2 = f(x2)

print("\nЗначения функции в узлах:")
print(f"f(x1) = {y1:.6f}")
print(f"f(x2) = {y2:.6f}")

# Базисные полиномы Лагранжа
def L1(x):
    return (x - x2)/(x1 - x2)

def L2(x):
    return (x - x1)/(x2 - x1)

# Полином Гаусса
def G(x):
    return y1 * L1(x) + y2 * L2(x)

# Значение в точке x0
exact_value = f(x0)
gauss_value = G(x0)
error = abs(exact_value - gauss_value)

print(f"\nРЕЗУЛЬТАТ в точке x0 = {x0}:")
print(f"Точное значение:  {exact_value:.6f}")
print(f"По Гауссу (n=2):  {gauss_value:.6f}")
print(f"Погрешность:      {error:.6f}")


# График 
x_plot = np.linspace(a, b, 100)
y_exact = f(x_plot)
y_gauss = [G(x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_exact, 'b-', linewidth=2, label='cos(sin²x) - точная')
plt.plot(x_plot, y_gauss, 'r--', linewidth=2, label='Полином Гаусса (n=2)')
plt.scatter([x1, x2], [y1, y2], color='green', s=80, zorder=5, label='Узлы Гаусса')
plt.scatter([x0], [exact_value], color='black', s=100, zorder=5, label=f'x0 = {x0}')
plt.axvline(x=x0, color='gray', linestyle=':', alpha=0.7)

plt.title('Интерполяция полиномом Гаусса (n=2)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
