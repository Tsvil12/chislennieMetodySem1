import math

def f(x):
    
    return math.cos(5 * x**2)

def trapezoidal_integral(a, b, n):
    
    h = (b - a) / n
    total = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        x = a + i * h
        total += f(x)
    
    return total * h

def simpson_integral(a, b, n):
    
    if n % 2 != 0:
        n += 1  # Делаем четным
    
    h = (b - a) / n
    total = f(a) + f(b)
    
    # Сумма нечетных точек
    for i in range(1, n, 2):
        x = a + i * h
        total += 4 * f(x)
    
    # Сумма четных точек
    for i in range(2, n-1, 2):
        x = a + i * h
        total += 2 * f(x)
    
    return total * h / 3

def method_1_trapezoid_by_error_estimate(a, b, epsilon):
    

    n_points = 1000
    max_f2 = 0
    for i in range(n_points + 1):
        x = a + (b - a) * i / n_points
        f2 = abs(-10 * math.sin(5 * x**2) - 100 * x**2 * math.cos(5 * x**2))
        if f2 > max_f2:
            max_f2 = f2

    n = 1
    while True:
        h = (b - a) / n
        R = (b - a)**3 / (12 * n**2) * max_f2
        if R <= epsilon:
            break
        n += 1
    
    # Вычисляем интеграл
    integral = trapezoidal_integral(a, b, n)
    
    return integral, n, R

def method_2_doubling_steps(a, b, epsilon, method='trapezoid'):
    
    if method == 'trapezoid':
        integrate_func = trapezoidal_integral
    else:  # simpson
        integrate_func = simpson_integral
    
    n = 2 if method == 'simpson' else 1  # Для Симпсона начинаем с четного
    prev_integral = integrate_func(a, b, n)
    
    while True:
        n *= 2
        current_integral = integrate_func(a, b, n)
        
        error = abs(current_integral - prev_integral)
        
        if error <= epsilon:
            break
            
        prev_integral = current_integral
    
    return current_integral, n, error

def main():
    a = 0
    b = 5
    epsilon = 1e-6
    
    print("="*60)
    print(f"Вычисление с точностью ε = {epsilon}")
    print("="*60)
    
    print("\nСПОСОБ 1: Выбор шага из оценки остаточного члена (трапеции)")
    I1, n1, R1 = method_1_trapezoid_by_error_estimate(a, b, epsilon)
    print(f"  Количество отрезков: {n1}")
    print(f"  Приближенное значение: {I1:.10f}")
    print(f"  Оценка погрешности: {R1:.2e}")
    
    print("\nСПОСОБ 2: Последовательное удвоение шагов (трапеции)")
    I2, n2, error2 = method_2_doubling_steps(a, b, epsilon, 'trapezoid')
    print(f"  Количество отрезков: {n2}")
    print(f"  Приближенное значение: {I2:.10f}")
    print(f"  Разница при удвоении: {error2:.2e}")
    
    print("\nСПОСОБ 2: Последовательное удвоение шагов (Симпсон)")
    I3, n3, error3 = method_2_doubling_steps(a, b, epsilon, 'simpson')
    print(f"  Количество отрезков: {n3}")
    print(f"  Приближенное значение: {I3:.10f}")
    print(f"  Разница при удвоении: {error3:.2e}")
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*60)
    print(f"Метод трапеций (способ 1): {I1:.10f}, n={n1}")
    print(f"Метод трапеций (способ 2): {I2:.10f}, n={n2}")
    print(f"Метод Симпсона (способ 2): {I3:.10f}, n={n3}")
    
    print(f"\nРазница трапеций (сп.1 - сп.2): {abs(I1 - I2):.2e}")
    print(f"Разница трапеций - Симпсон: {abs(I2 - I3):.2e}")

if __name__ == "__main__":
    main()