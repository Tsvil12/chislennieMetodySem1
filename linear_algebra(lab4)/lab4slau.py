def steepest_descent_method(A, f, eps=1e-6, max_iter=10000):
    """Явный метод скорейшего спуска (минимальных невязок) для Ax = f."""
    n = len(A)
    x = [0.0] * n
    r = [f[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

    for k in range(max_iter):
        Ar = [sum(A[i][j] * r[j] for j in range(n)) for i in range(n)]
        num = sum(r[i] * r[i] for i in range(n))
        den = sum(r[i] * Ar[i] for i in range(n))

        if abs(den) < 1e-12:
            break
        tau = num / den

        for i in range(n):
            x[i] += tau * r[i]
            r[i] -= tau * Ar[i]

        norm_r = max(abs(r[i]) for i in range(n))
        if norm_r < eps:
            return x, k + 1
    return x, max_iter


def minimal_residual_method(A, f, eps=1e-6, max_iter=10000):
    """Метод минимальных невязок (явный)."""
    n = len(A)
    x = [0.0] * n
    r = [f[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

    for k in range(max_iter):
        Ar = [sum(A[i][j] * r[j] for j in range(n)) for i in range(n)]
        num = sum(r[i] * Ar[i] for i in range(n))
        den = sum(Ar[i] * Ar[i] for i in range(n))

        if abs(den) < 1e-12:
            break
        tau = num / den

        for i in range(n):
            x[i] += tau * r[i]
            r[i] -= tau * Ar[i]

        norm_r = max(abs(r[i]) for i in range(n))
        if norm_r < eps:
            return x, k + 1
    return x, max_iter


def minimal_correction_method(A, f, eps=1e-6, max_iter=10000):
    """Метод минимальных поправок (неявный)."""
    n = len(A)
    x = [0.0] * n
    r = [f[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    p = [sum(A[i][j] * r[j] for j in range(n)) for i in range(n)]

    for k in range(max_iter):
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        num = sum(r[i] * p[i] for i in range(n))
        den = sum(p[i] * Ap[i] for i in range(n))

        if abs(den) < 1e-12:
            break
        tau = num / den

        for i in range(n):
            x[i] += tau * p[i]
            r[i] -= tau * Ap[i]

        p = [sum(A[i][j] * r[j] for j in range(n)) for i in range(n)]

        norm_r = max(abs(r[i]) for i in range(n))
        if norm_r < eps:
            return x, k + 1
    return x, max_iter


def minimal_error_method(A, f, eps=1e-6, max_iter=10000):
    """Метод минимальных погрешностей (сопряжённые градиенты)."""
    n = len(A)
    x = [0.0] * n
    r = [f[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    z = r[:]

    for k in range(max_iter):
        Az = [sum(A[i][j] * z[j] for j in range(n)) for i in range(n)]
        num = sum(r[i] * r[i] for i in range(n))
        den = sum(z[i] * Az[i] for i in range(n))

        if abs(den) < 1e-12:
            break
        alpha = num / den

        for i in range(n):
            x[i] += alpha * z[i]
            r[i] -= alpha * Az[i]

        beta = sum(r[i] * r[i] for i in range(n)) / num

        for i in range(n):
            z[i] = r[i] + beta * z[i]

        norm_r = max(abs(r[i]) for i in range(n))
        if norm_r < eps:
            return x, k + 1
    return x, max_iter


# ============ ТЕСТИРОВАНИЕ ============
# Пример: 2x2 система
# 4x1 + x2 = 1
# x1 + 3x2 = 2

A = [[4.0, 1.0],
     [1.0, 3.0]]
f = [1.0, 2.0]

methods = [
    ("Скорейший спуск", steepest_descent_method),
    ("Мин. невязок", minimal_residual_method),
    ("Мин. поправок", minimal_correction_method),
    ("Мин. погрешностей", minimal_error_method)
]

print("Метод".ljust(20), "Итерации", "Решение")
print("-" * 50)
for name, method in methods:
    sol, it = method(A, f, eps=1e-8)
    print(f"{name:<20} {it:4}   {sol}")