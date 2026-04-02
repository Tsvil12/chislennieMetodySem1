# ============================================================================
# НАХОЖДЕНИЕ ОБРАТНОЙ МАТРИЦЫ МЕТОДОМ ГАУССА
# Методичка: стр. 38
# 
# Суть: решаем n систем A * x^k = e^k, где e^k - единичные векторы
# Полученные решения x^k - это столбцы обратной матрицы
# ============================================================================

def gauss_solve(A, f):
    """
    Решение СЛАУ Ax = f (вспомогательная функция)
    """
    n = len(A)
    a = [row[:] for row in A]
    b = f[:]
    
    # Прямой ход
    for k in range(n):
        # Выбор главного элемента
        max_row = k
        for i in range(k + 1, n):
            if abs(a[i][k]) > abs(a[max_row][k]):
                max_row = i
        if max_row != k:
            a[k], a[max_row] = a[max_row], a[k]
            b[k], b[max_row] = b[max_row], b[k]
        
        # Нормализация
        diag = a[k][k]
        for j in range(k, n):
            a[k][j] = a[k][j] / diag
        b[k] = b[k] / diag
        
        # Исключение
        for i in range(k + 1, n):
            factor = a[i][k]
            for j in range(k, n):
                a[i][j] = a[i][j] - factor * a[k][j]
            b[i] = b[i] - factor * b[k]
    
    # Обратный ход
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] = x[i] - a[i][j] * x[j]
    return x


def inverse_matrix(A):
    """
    Нахождение обратной матрицы A^{-1}.
    
    ПРИМЕР:
    A = [[4, 1, 1],
         [1, 5, 1],
         [1, 1, 6]]
    
    Обратная матрица:
    [[ 0.5446, -0.0446, -0.0446],
     [-0.0446,  0.5446, -0.0446],
     [-0.0446, -0.0446,  0.5446]]
    """
    n = len(A)
    
    # Создаём пустую матрицу для результата
    A_inv = [[0] * n for _ in range(n)]
    
    # Для каждого столбца обратной матрицы
    for k in range(n):
        # Создаём единичный вектор e_k
        e = [0] * n
        e[k] = 1
        
        # Решаем систему A * x = e
        x = gauss_solve(A, e)
        
        # Записываем решение в k-й столбец обратной матрицы
        for i in range(n):
            A_inv[i][k] = x[i]
    
    return A_inv


def check_inverse(A, A_inv):
    """
    Проверка: A * A^{-1} должна быть равна единичной матрице E.
    """
    n = len(A)
    # Умножаем A на A_inv
    product = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                product[i][j] = product[i][j] + A[i][k] * A_inv[k][j]
    
    # Сравниваем с единичной матрицей
    max_error = 0
    for i in range(n):
        for j in range(n):
            expected = 1 if i == j else 0
            error = abs(product[i][j] - expected)
            if error > max_error:
                max_error = error
    
    return max_error < 1e-10, max_error


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========
if __name__ == "__main__":
    A = [
        [4, 1, 1],
        [1, 5, 1],
        [1, 1, 6]
    ]
    
    print("Исходная матрица A:")
    for row in A:
        print(f"  {row}")
    
    A_inv = inverse_matrix(A)
    
    print("\nОбратная матрица A^{-1}:")
    for row in A_inv:
        print(f"  {[round(x, 4) for x in row]}")
    
    # Проверка
    is_ok, error = check_inverse(A, A_inv)
    print(f"\nПроверка A * A^{-1} = E: {'ДА' if is_ok else 'НЕТ'}")
    print(f"Максимальная ошибка: {error:.2e}")