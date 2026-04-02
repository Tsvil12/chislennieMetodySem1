import numpy as np
import matplotlib.pyplot as plt
from math import cos

class NonlinearEquationSolver:
    """
    Класс для решения нелинейного уравнения 4x^4 - 6.2 - cos(0.6x) = 0
    различными численными методами
    """
    
    def __init__(self, eps=1e-6, max_iter=1000):
        """
        Инициализация решателя
        
        Параметры:
        eps - точность вычислений
        max_iter - максимальное число итераций
        """
        self.eps = eps
        self.max_iter = max_iter
        
    def f(self, x):
        """Исходная функция f(x) = 4x^4 - 6.2 - cos(0.6x)"""
        return 4 * x**4 - 6.2 - cos(0.6 * x)
    
    def f_prime(self, x):
        """Первая производная f'(x) = 16x^3 + 0.6*sin(0.6x)"""
        return 16 * x**3 + 0.6 * np.sin(0.6 * x)
    
    def f_double_prime(self, x):
        """Вторая производная f''(x) = 48x^2 + 0.36*cos(0.6x)"""
        return 48 * x**2 + 0.36 * np.cos(0.6 * x)
    
    def g(self, x):
        """
        Функция для метода простой итерации x = g(x)
        Преобразуем исходное уравнение: x = ( (6.2 + cos(0.6x))/4 )^(1/4)
        """
        return ((6.2 + np.cos(0.6 * x)) / 4) ** 0.25
    
    def g_prime(self, x):
        """Производная функции g(x) для проверки условия сходимости"""
        # Аналитическое выражение сложное, используем численное дифференцирование
        h = 1e-7
        return (self.g(x + h) - self.g(x)) / h
    
    def separate_roots(self, a=-5, b=5, num_points=1000):
        """
        Отделение корней графическим и аналитическим способом
        
        Параметры:
        a, b - границы интервала поиска
        num_points - количество точек для построения графика
        
        Возвращает:
        roots_intervals - список интервалов, содержащих корни
        """
        x = np.linspace(a, b, num_points)
        y = [self.f(xi) for xi in x]
        
        # Поиск интервалов, где функция меняет знак
        roots_intervals = []
        for i in range(len(x) - 1):
            if y[i] * y[i + 1] < 0:
                roots_intervals.append((x[i], x[i + 1]))
        
        # Построение графика
        plt.figure(figsize=(12, 8))
        plt.plot(x, y, 'b-', linewidth=2, label='f(x) = 4x⁴ - 6.2 - cos(0.6x)')
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='y=0')
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Графическое отделение корней уравнения\n4x⁴ - 6.2 - cos(0.6x) = 0')
        plt.legend()
        
        # Отметить найденные корни
        for interval in roots_intervals:
            plt.axvspan(interval[0], interval[1], alpha=0.2, color='green')
            
        plt.show()
        
        # Аналитическая проверка
        print("\n" + "="*60)
        print("ОТДЕЛЕНИЕ КОРНЕЙ")
        print("="*60)
        print(f"На интервале [{a}, {b}] найдено {len(roots_intervals)} интервалов, содержащих корни:")
        for i, interval in enumerate(roots_intervals):
            f_a = self.f(interval[0])
            f_b = self.f(interval[1])
            print(f"\nКорень {i+1}:")
            print(f"  Интервал: [{interval[0]:.4f}, {interval[1]:.4f}]")
            print(f"  f({interval[0]:.4f}) = {f_a:.6f}")
            print(f"  f({interval[1]:.4f}) = {f_b:.6f}")
            print(f"  f(a) * f(b) = {f_a * f_b:.6f} < 0 - корень существует")
            
            # Проверка монотонности
            x_mid = (interval[0] + interval[1]) / 2
            f_prime_mid = self.f_prime(x_mid)
            f_double_prime_mid = self.f_double_prime(x_mid)
            print(f"  f'({x_mid:.4f}) = {f_prime_mid:.6f} - {'возрастает' if f_prime_mid > 0 else 'убывает'}")
            print(f"  f''({x_mid:.4f}) = {f_double_prime_mid:.6f} - {'выпукла вниз' if f_double_prime_mid > 0 else 'выпукла вверх'}")
        
        return roots_intervals
    
    def bisection_method(self, a, b, verbose=True):
        """
        Метод дихотомии (половинного деления)
        
        Параметры:
        a, b - границы интервала [a, b]
        verbose - вывод промежуточных результатов
        
        Возвращает:
        root - найденный корень
        iterations - количество итераций
        history - история приближений
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД ДИХОТОМИИ (ПОЛОВИННОГО ДЕЛЕНИЯ)")
            print("="*60)
        
        history = []
        iterations = 0
        
        # Проверка наличия корня на интервале
        if self.f(a) * self.f(b) > 0:
            raise ValueError("На интервале нет корня или их четное количество")
        
        while iterations < self.max_iter:
            c = (a + b) / 2
            fc = self.f(c)
            history.append((iterations, c, fc, b-a))
            
            if verbose and iterations % 5 == 0:
                print(f"Итерация {iterations}: x = {c:.10f}, f(x) = {fc:.10e}, длина интервала = {b-a:.10e}")
            
            # Проверка условий остановки
            if abs(fc) < self.eps or (b - a) / 2 < self.eps:
                break
                
            if self.f(a) * fc < 0:
                b = c
            else:
                a = c
                
            iterations += 1
        
        root = (a + b) / 2
        if verbose:
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Корень x = {root:.10f}")
            print(f"Значение f(x) = {self.f(root):.10e}")
            print(f"Количество итераций: {iterations}")
            print(f"Погрешность: {(b-a)/2:.10e}")
        
        return root, iterations, history
    
    def simple_iteration_method(self, initial_x, verbose=True):
        """
        Метод простой итерации
        
        Параметры:
        initial_x - начальное приближение
        verbose - вывод промежуточных результатов
        
        Возвращает:
        root - найденный корень
        iterations - количество итераций
        history - история приближений
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
            print("="*60)
        
        x_prev = initial_x
        history = [(0, x_prev, self.f(x_prev))]
        iterations = 0
        
        # Проверка условия сходимости
        q = abs(self.g_prime(initial_x))
        if q >= 1:
            print(f"ВНИМАНИЕ: |g'(x0)| = {q:.6f} >= 1, сходимость не гарантируется")
        
        if verbose:
            print(f"Начальное приближение: x0 = {initial_x}")
            print(f"|g'(x0)| = {q:.6f} {'< 1 - условие сходимости выполняется' if q < 1 else '>= 1 - условие сходимости НЕ выполняется'}")
        
        while iterations < self.max_iter:
            x_next = self.g(x_prev)
            fx = self.f(x_next)
            iterations += 1
            history.append((iterations, x_next, fx))
            
            if verbose and iterations % 5 == 0:
                print(f"Итерация {iterations}: x = {x_next:.10f}, f(x) = {fx:.10e}, |x_new - x_old| = {abs(x_next - x_prev):.10e}")
            
            # Критерий остановки с учетом скорости сходимости
            if q < 1:
                if abs(x_next - x_prev) <= (1 - q) * self.eps / q:
                    break
            else:
                if abs(x_next - x_prev) < self.eps:
                    break
            
            x_prev = x_next
        
        root = x_next
        if verbose:
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Корень x = {root:.10f}")
            print(f"Значение f(x) = {self.f(root):.10e}")
            print(f"Количество итераций: {iterations}")
        
        return root, iterations, history
    
    def newton_method(self, initial_x, verbose=True):
        """
        Метод Ньютона (касательных)
        
        Параметры:
        initial_x - начальное приближение
        verbose - вывод промежуточных результатов
        
        Возвращает:
        root - найденный корень
        iterations - количество итераций
        history - история приближений
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД НЬЮТОНА (КАСАТЕЛЬНЫХ)")
            print("="*60)
        
        x = initial_x
        history = [(0, x, self.f(x))]
        iterations = 0
        
        if verbose:
            print(f"Начальное приближение: x0 = {x}")
        
        while iterations < self.max_iter:
            f_val = self.f(x)
            f_prime_val = self.f_prime(x)
            
            if abs(f_prime_val) < 1e-12:
                raise ValueError(f"Производная близка к нулю: f'({x}) = {f_prime_val}")
            
            x_next = x - f_val / f_prime_val
            iterations += 1
            history.append((iterations, x_next, self.f(x_next)))
            
            if verbose and iterations % 5 == 0:
                print(f"Итерация {iterations}: x = {x_next:.10f}, f(x) = {self.f(x_next):.10e}, |x_new - x_old| = {abs(x_next - x):.10e}")
            
            if abs(x_next - x) < self.eps:
                break
                
            x = x_next
        
        root = x
        if verbose:
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Корень x = {root:.10f}")
            print(f"Значение f(x) = {self.f(root):.10e}")
            print(f"Количество итераций: {iterations}")
        
        return root, iterations, history
    
    def secant_method(self, a, b, verbose=True):
        """
        Метод хорд (секущих)
        
        Параметры:
        a, b - границы интервала [a, b]
        verbose - вывод промежуточных результатов
        
        Возвращает:
        root - найденный корень
        iterations - количество итераций
        history - история приближений
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД ХОРД")
            print("="*60)
        
        # Определяем неподвижный конец
        if self.f(a) * self.f_double_prime(a) > 0:
            fixed = a
            x0 = b
            print(f"Неподвижный конец: a = {a} (f(a)*f''(a) > 0)")
        else:
            fixed = b
            x0 = a
            print(f"Неподвижный конец: b = {b} (f(b)*f''(b) > 0)")
        
        x_prev = x0
        history = [(0, x_prev, self.f(x_prev))]
        iterations = 0
        
        if verbose:
            print(f"Начальное приближение: x0 = {x0}")
        
        while iterations < self.max_iter:
            f_fixed = self.f(fixed)
            f_prev = self.f(x_prev)
            
            if abs(f_prev - f_fixed) < 1e-12:
                raise ValueError("Разность значений функции близка к нулю")
            
            x_next = x_prev - f_prev * (x_prev - fixed) / (f_prev - f_fixed)
            iterations += 1
            history.append((iterations, x_next, self.f(x_next)))
            
            if verbose and iterations % 5 == 0:
                print(f"Итерация {iterations}: x = {x_next:.10f}, f(x) = {self.f(x_next):.10e}, |x_new - x_old| = {abs(x_next - x_prev):.10e}")
            
            if abs(x_next - x_prev) < self.eps:
                break
                
            x_prev = x_next
        
        root = x_next
        if verbose:
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Корень x = {root:.10f}")
            print(f"Значение f(x) = {self.f(root):.10e}")
            print(f"Количество итераций: {iterations}")
        
        return root, iterations, history
    
    def chebyshev_method(self, initial_x, verbose=True):
        """
        Метод Чебышева (третьего порядка точности)
        
        Параметры:
        initial_x - начальное приближение
        verbose - вывод промежуточных результатов
        
        Возвращает:
        root - найденный корень
        iterations - количество итераций
        history - история приближений
        """
        if verbose:
            print("\n" + "="*60)
            print("МЕТОД ЧЕБЫШЕВА")
            print("="*60)
        
        x = initial_x
        history = [(0, x, self.f(x))]
        iterations = 0
        
        if verbose:
            print(f"Начальное приближение: x0 = {x}")
        
        while iterations < self.max_iter:
            f_val = self.f(x)
            f_prime_val = self.f_prime(x)
            f_double_prime_val = self.f_double_prime(x)
            
            if abs(f_prime_val) < 1e-12:
                raise ValueError(f"Производная близка к нулю: f'({x}) = {f_prime_val}")
            
            # Формула Чебышева: x_{n+1} = x_n - f/f' - (f'' * f^2)/(2 * (f')^3)
            x_next = x - f_val / f_prime_val - (f_double_prime_val * f_val**2) / (2 * f_prime_val**3)
            iterations += 1
            history.append((iterations, x_next, self.f(x_next)))
            
            if verbose and iterations % 5 == 0:
                print(f"Итерация {iterations}: x = {x_next:.10f}, f(x) = {self.f(x_next):.10e}, |x_new - x_old| = {abs(x_next - x):.10e}")
            
            if abs(x_next - x) < self.eps:
                break
                
            x = x_next
        
        root = x
        if verbose:
            print(f"\nРЕЗУЛЬТАТ:")
            print(f"Корень x = {root:.10f}")
            print(f"Значение f(x) = {self.f(root):.10e}")
            print(f"Количество итераций: {iterations}")
        
        return root, iterations, history
    
    def verify_root(self, root):
        """
        Проверка корректности найденного корня
        
        Параметры:
        root - найденный корень
        
        Возвращает:
        dict с результатами проверки
        """
        verification = {
            'root': root,
            'f(root)': self.f(root),
            '|f(root)|': abs(self.f(root)),
            'condition': 'OK' if abs(self.f(root)) < self.eps else 'ПРЕДУПРЕЖДЕНИЕ',
            'f_prime(root)': self.f_prime(root)
        }
        
        # Проверка знака функции вокруг корня
        delta = self.eps * 10
        left = root - delta
        right = root + delta
        f_left = self.f(left)
        f_right = self.f(right)
        
        verification['f(left)'] = f_left
        verification['f(right)'] = f_right
        verification['sign_change'] = f_left * f_right < 0
        
        return verification
    
    def compare_methods(self, intervals, initial_guesses):
        """
        Сравнение всех методов
        
        Параметры:
        intervals - интервалы для методов, требующих интервал
        initial_guesses - начальные приближения для методов, требующих начальную точку
        """
        print("\n" + "="*60)
        print("СРАВНЕНИЕ МЕТОДОВ РЕШЕНИЯ")
        print("="*60)
        
        results = {}
        
        for i, interval in enumerate(intervals):
            print(f"\n--- КОРЕНЬ {i+1} на интервале [{interval[0]:.4f}, {interval[1]:.4f}] ---")
            
            # Метод дихотомии
            root_bis, iter_bis, _ = self.bisection_method(interval[0], interval[1], verbose=False)
            results['Дихотомия'] = {'root': root_bis, 'iterations': iter_bis}
            
            # Метод хорд
            root_sec, iter_sec, _ = self.secant_method(interval[0], interval[1], verbose=False)
            results['Хорд'] = {'root': root_sec, 'iterations': iter_sec}
            
            # Метод простой итерации
            root_simp, iter_simp, _ = self.simple_iteration_method(initial_guesses[i], verbose=False)
            results['Простая итерация'] = {'root': root_simp, 'iterations': iter_simp}
            
            # Метод Ньютона
            root_newt, iter_newt, _ = self.newton_method(initial_guesses[i], verbose=False)
            results['Ньютона'] = {'root': root_newt, 'iterations': iter_newt}
            
            # Метод Чебышева
            root_cheb, iter_cheb, _ = self.chebyshev_method(initial_guesses[i], verbose=False)
            results['Чебышева'] = {'root': root_cheb, 'iterations': iter_cheb}
            
            # Вывод результатов сравнения
            print(f"{'Метод':<20} {'Найденный корень':<25} {'Итерации':<10} {'|f(x)|':<15}")
            print("-"*70)
            for method, data in results.items():
                print(f"{method:<20} {data['root']:<25.10f} {data['iterations']:<10} {abs(self.f(data['root'])):<15.10e}")
            
            # Проверка корректности для одного из корней
            print(f"\nПРОВЕРКА КОРРЕКТНОСТИ ДЛЯ КОРНЯ {i+1} (метод Ньютона):")
            verification = self.verify_root(results['Ньютона']['root'])
            for key, value in verification.items():
                print(f"  {key}: {value}")


def main():
    """Основная функция для решения задачи"""
    
    print("="*80)
    print("РЕШЕНИЕ НЕЛИНЕЙНОГО УРАВНЕНИЯ")
    print("f(x) = 4x⁴ - 6.2 - cos(0.6x) = 0")
    print("="*80)
    
    # Создание решателя
    solver = NonlinearEquationSolver(eps=1e-8, max_iter=1000)
    
    # 1. Отделение корней
    intervals = solver.separate_roots(a=-5, b=5)
    
    if len(intervals) == 0:
        print("Корни не найдены. Проверьте интервал поиска.")
        return
    
    # 2. Уточнение корней всеми методами
    # Начальные приближения для методов, требующих начальную точку
    initial_guesses = [(interval[0] + interval[1]) / 2 for interval in intervals]
    
    # Демонстрация работы методов для первого корня
    print("\n" + "="*80)
    print("ДЕМОНСТРАЦИЯ РАБОТЫ МЕТОДОВ ДЛЯ ПЕРВОГО КОРНЯ")
    print("="*80)
    
    interval1 = intervals[0]
    guess1 = initial_guesses[0]
    
    print(f"\nИнтервал: [{interval1[0]:.4f}, {interval1[1]:.4f}]")
    print(f"Начальное приближение: {guess1:.4f}")
    
    # Дихотомия
    root1, iter1, hist1 = solver.bisection_method(interval1[0], interval1[1])
    
    # Простая итерация
    root2, iter2, hist2 = solver.simple_iteration_method(guess1)
    
    # Ньютон
    root3, iter3, hist3 = solver.newton_method(guess1)
    
    # Хорды
    root4, iter4, hist4 = solver.secant_method(interval1[0], interval1[1])
    
    # Чебышев
    root5, iter5, hist5 = solver.chebyshev_method(guess1)
    
    # 3. Сравнение всех методов для всех корней
    solver.compare_methods(intervals, initial_guesses)
    
    # 4. Ответы на теоретические вопросы
    print("\n" + "="*80)
    print("ОТВЕТЫ НА ТЕОРЕТИЧЕСКИЕ ВОПРОСЫ")
    print("="*80)
    
    print("\n1. КАК БРАТЬ НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ КОРНЯ?")
    print("   - Для методов, требующих интервал (дихотомия, хорды): выбирается интервал [a,b],")
    print("     на котором f(a)*f(b) < 0 (функция меняет знак).")
    print("   - Для методов, требующих начальную точку (итераций, Ньютона, Чебышева):")
    print("     выбирается точка внутри интервала, содержащего корень. Желательно выбирать")
    print("     точку, где производная не равна нулю и знак функции совпадает со знаком")
    print("     второй производной для гарантии сходимости.")
    print("   - В данной работе начальное приближение взято как середина интервала.")
    
    print("\n2. КАК ОСТАНОВИТЬ ПРОЦЕСС УТОЧНЕНИЯ КОРНЯ?")
    print("   Используются следующие критерии остановки:")
    print("   - |x_{n+1} - x_n| < ε (малость изменения приближения)")
    print("   - |f(x_n)| < ε (близость к нулю значения функции)")
    print("   - Для метода дихотомии: длина интервала < 2ε")
    print("   - Для метода простой итерации с q < 1: |x_{n+1} - x_n| ≤ (1-q)ε/q")
    print("   - Также ограничивается максимальное число итераций")
    
    print("\n3. КАК ПРОВЕРИТЬ ПРАВИЛЬНОСТЬ НАХОЖДЕНИЯ КОРНЯ?")
    print("   - Подстановка найденного корня в исходное уравнение: |f(x*)| < ε")
    print("   - Проверка знака функции слева и справа от корня")
    print("   - Сравнение результатов, полученных разными методами")
    print("   - Проверка устойчивости: небольшое изменение начального приближения")
    print("     должно приводить к близкому результату")
    
    print("\n4. ПРЕИМУЩЕСТВА И ОГРАНИЧЕНИЯ МЕТОДОВ:")
    print("   МЕТОД ДИХОТОМИИ:")
    print("     + Гарантированная сходимость")
    print("     + Простота реализации")
    print("     + Не требует вычисления производных")
    print("     - Медленная сходимость (линейная)")
    print("     - Требует интервал со сменой знака")
    
    print("\n   МЕТОД ПРОСТОЙ ИТЕРАЦИИ:")
    print("     + Простота реализации")
    print("     + Не требует вычисления производных")
    print("     - Сходимость зависит от выбора функции g(x)")
    print("     - Требует выполнения условия |g'(x)| < 1")
    print("     - Линейная сходимость")
    
    print("\n   МЕТОД НЬЮТОНА:")
    print("     + Быстрая сходимость (квадратичная)")
    print("     + Не требует интервала со сменой знака")
    print("     - Требует вычисления производной")
    print("     - Может расходиться при плохом начальном приближении")
    print("     - Может зациклиться, если производная близка к нулю")
    
    print("\n   МЕТОД ХОРД:")
    print("     + Не требует вычисления производной")
    print("     + Проще метода Ньютона")
    print("     - Линейная сходимость")
    print("     - Требует интервала со сменой знака")
    print("     - Может быть медленнее метода дихотомии на некоторых функциях")
    
    print("\n   МЕТОД ЧЕБЫШЕВА:")
    print("     + Очень быстрая сходимость (кубическая)")
    print("     - Требует вычисления первой и второй производных")
    print("     - Сложнее в реализации")
    print("     - Чувствителен к выбору начального приближения")
    
    print("\n5. КОЛИЧЕСТВО ИТЕРАЦИЙ ПРИ eps = 1e-8:")
    print(f"   Метод дихотомии: {iter1} итераций")
    print(f"   Метод простой итерации: {iter2} итераций")
    print(f"   Метод Ньютона: {iter3} итераций")
    print(f"   Метод хорд: {iter4} итераций")
    print(f"   Метод Чебышева: {iter5} итераций")
    
    print("\n   Вывод: методы Ньютона и Чебышева сходятся быстрее всего,")
    print("   но требуют вычисления производных. Метод дихотомии самый")
    print("   надежный, но медленный. Метод простой итерации может быть")
    print("   быстрее дихотомии при удачном выборе функции g(x).")


if __name__ == "__main__":
    main()