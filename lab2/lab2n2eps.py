import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.cos(np.sin(x)**2)

a = 0
b = np.pi
x0 = 1.5  
epsilon = 1e-10

print(f"Функция: y = cos(sin²x)")
print(f"Отрезок: [{a}, {b}]")
print(f"Точка x0 = {x0}")
print(f"Точность ε = {epsilon}\n")

def lagrange_nodes(n, interval_start=a, interval_end=b):
    """Возвращает равномерно распределённые узлы Лагранжа."""
    return np.linspace(interval_start, interval_end, n)

def interpolate_lagrange(x_values, y_values, x):
    """Классический интерполяционный полином Лагранжа."""
    result = 0
    for i in range(len(y_values)):
        product = 1
        for j in range(len(y_values)):
            if i != j:
                product *= (x - x_values[j]) / (x_values[i] - x_values[j])
        result += y_values[i] * product
    return result

n = 2
iteration = 0
max_iterations = 10  

while True:
    iteration += 1
    
    # Узлы Лагранжа 
    x_nodes = lagrange_nodes(n)
    
    # Вычисляем значения функции в узлах
    y_nodes = [f(xi) for xi in x_nodes]
    
    print(f"\nИтерация №{iteration}, n={n}:")
    print("Узлы Лагранжа:", ", ".join(map(lambda x: f"{x:.6f}", x_nodes)))
    print("Значения функции в узлах:", ", ".join(map(lambda y: f"{y:.6f}", y_nodes)))
    
    # Интерполяция в точке x0
    exact_value = f(x0)
    approx_value = interpolate_lagrange(x_nodes, y_nodes, x0)
    error = abs(exact_value - approx_value)
    
    print(f"Приближенное значение в x0: {approx_value:.6f}")
    print(f"Ошибка: {error:.6f}")
    
    # Проверка условия остановки
    if iteration >= max_iterations:
        break
        
    # Если точность недостаточна, увеличиваем порядок полинома
    n += 1

x_plot = np.linspace(a, b, 100)
y_exact = f(x_plot)
y_interpolated = [interpolate_lagrange(x_nodes, y_nodes, x) for x in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_exact, 'b-', linewidth=2, label='Точная функция')
plt.plot(x_plot, y_interpolated, 'r--', linewidth=2, label=f'Полином Лагранжа (n={n})')
plt.scatter(x_nodes, y_nodes, color='green', s=80, zorder=5, label='Узлы Лагранжа')
plt.scatter([x0], [exact_value], color='black', s=100, zorder=5, label=f'x0 = {x0}')
plt.axvline(x=x0, color='gray', linestyle=':', alpha=0.7)

plt.title('Интерполяция полиномом Лагранжа')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()