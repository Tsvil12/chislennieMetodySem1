# ============================================================================
# МЕТОД ЯКОБИ (МЕТОД ПРОСТОЙ ИТЕРАЦИИ)
# Методичка: стр. 39
# 
# Суть: выражаем x_i из i-го уравнения и итерационно уточняем.
# Формула: x_i^{(k+1)} = (f_i - сумма a_ij * x_j^{(k)}) / a_ii
# ============================================================================


"""
    Решение СЛАУ Ax = f методом Якоби.
    
    ПРИМЕР:
    A = [[4, 1, 1],
         [1, 5, 1],
         [1, 1, 6]]
    f = [9, 14, 21]
    
    Ответ: x = [1, 2, 3]
    """
    
    
def jacobi(A, f, epsilon=1e-6, max_iter=1000):
    n = len(A)
    
    # Начальное приближение (берём нули)
    x = [0] * n
    x_new = [0] * n
    
    for iter_num in range(max_iter):
        # Вычисляем новое приближение
        for i in range(n):
            # Сумма a_ij * x_j для j != i
            s = 0
            for j in range(n):
                if j != i:
                    s = s + A[i][j] * x[j]
            # Формула Якоби: x_i = (f_i - сумма) / a_ii
            x_new[i] = (f[i] - s) / A[i][i]
        
        # Проверяем, достигнута ли точность
        # Считаем максимальное изменение
        max_change = 0
        for i in range(n):
            change = abs(x_new[i] - x[i])
            if change > max_change:
                max_change = change
        
        # Обновляем x
        x = x_new[:]
        
        # Если изменения маленькие - останавливаемся
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
    
    x = jacobi(A, f)
    
    print(f"\nРешение: x = {[round(val, 6) for val in x]}")
    
    # Проверка
    print("\nПроверка A*x:")
    for i in range(len(A)):
        val = sum(A[i][j] * x[j] for j in range(len(A)))
        print(f"  Строка {i}: {val:.6f} (должно быть {f[i]})")