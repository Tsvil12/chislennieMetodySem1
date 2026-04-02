# ============================================================================
# МЕТОД ЗЕЙДЕЛЯ
# Методичка: стр. 40
# 
# Суть: как метод Якоби, но новые значения используем сразу.
# Это ускоряет сходимость.
# ============================================================================

def seidel(A, f, epsilon=1e-6, max_iter=1000):
    """
    Решение СЛАУ Ax = f методом Зейделя.
    
    ПРИМЕР:
    A = [[4, 1, 1],
         [1, 5, 1],
         [1, 1, 6]]
    f = [9, 14, 21]
    
    Ответ: x = [1, 2, 3]
    """
    n = len(A)
    
    # Начальное приближение
    x = [0] * n
    
    for iter_num in range(max_iter):
        x_old = x[:]  # Сохраняем старые значения для проверки
        
        # Обновляем каждую компоненту
        for i in range(n):
            # Сумма a_ij * x_j
            s = 0
            for j in range(n):
                if j != i:
                    s = s + A[i][j] * x[j]
            # Обновляем x_i (используем уже обновлённые x_j для j < i)
            x[i] = (f[i] - s) / A[i][i]
        
        # Проверяем сходимость
        max_change = 0
        for i in range(n):
            change = abs(x[i] - x_old[i])
            if change > max_change:
                max_change = change
        
        if max_change < epsilon:
            print(f"Сошелся за {iter_num + 1} итераций")
            return x
    
    print(f"Не сошелся за {max_iter} итераций")
    return x


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    A = [
        [4, 1, 1],
        [1, 5, 1],
        [1, 1, 6]
    ]
    f = [9, 14, 21]
    
    print("Матрица A:")
    for row in A:
        print(f"  {row}")
    print(f"Вектор f: {f}")
    
    x = seidel(A, f)
    
    print(f"\nРешение: x = {[round(val, 6) for val in x]}")
    
    # Проверка
    print("\nПроверка A*x:")
    for i in range(len(A)):
        val = sum(A[i][j] * x[j] for j in range(len(A)))
        print(f"  Строка {i}: {val:.6f} (должно быть {f[i]})")