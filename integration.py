import numpy as np


class NewtonCotes:
    """Regles composites de Newton-Cotes."""

    @staticmethod
    def rectangle(f, a, b, n=1):
        if n <= 0:
            raise ValueError("n doit etre strictement positif.")
        h = (b - a) / n
        x_mid = a + (np.arange(n) + 0.5) * h
        return float(h * np.sum(f(x_mid)))

    @staticmethod
    def trapezoidal(f, a, b, n=1):
        if n <= 0:
            raise ValueError("n doit etre strictement positif.")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return float(h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1]))

    @staticmethod
    def simpson(f, a, b, n=2):
        if n <= 0 or n % 2 != 0:
            raise ValueError("n doit etre strictement positif et pair pour Simpson.")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)
        return float((h / 3.0) * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))

    @staticmethod
    def simpson_38(f, a, b, n=3):
        if n <= 0 or n % 3 != 0:
            raise ValueError("n doit etre strictement positif et multiple de 3 pour Simpson 3/8.")
        h = (b - a) / n
        x = np.linspace(a, b, n + 1)
        y = f(x)

        idx = np.arange(1, n)
        mult3 = idx % 3 == 0
        sum3 = np.sum(y[idx[mult3]])
        sum_not3 = np.sum(y[idx[~mult3]])
        return float((3.0 * h / 8.0) * (y[0] + y[-1] + 2.0 * sum3 + 3.0 * sum_not3))


class AdaptiveIntegration:
    """Integration adaptative basee sur Simpson."""

    def __init__(self, tol=1e-6, max_depth=20):
        self.tol = float(tol)
        self.max_depth = int(max_depth)

    def _simpson_rule(self, f, a, b):
        c = 0.5 * (a + b)
        return (b - a) * (f(a) + 4.0 * f(c) + f(b)) / 6.0

    def adaptive_simpson(self, f, a, b, tol=None, depth=0):
        tol_local = self.tol if tol is None else float(tol)

        c = 0.5 * (a + b)
        s_ab = self._simpson_rule(f, a, b)
        s_ac = self._simpson_rule(f, a, c)
        s_cb = self._simpson_rule(f, c, b)

        err = abs((s_ac + s_cb) - s_ab)
        if depth >= self.max_depth or err < 15.0 * tol_local:
            return (s_ac + s_cb) + ((s_ac + s_cb) - s_ab) / 15.0

        left = self.adaptive_simpson(f, a, c, tol=0.5 * tol_local, depth=depth + 1)
        right = self.adaptive_simpson(f, c, b, tol=0.5 * tol_local, depth=depth + 1)
        return left + right


class GaussQuadrature:
    """Quadrature de Gauss-Legendre 2 et 3 points."""

    @staticmethod
    def gauss_legendre_2(f, a, b):
        nodes = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
        weights = np.array([1.0, 1.0])
        xm = 0.5 * (a + b)
        xr = 0.5 * (b - a)
        x_map = xm + xr * nodes
        return float(xr * np.sum(weights * f(x_map)))

    @staticmethod
    def gauss_legendre_3(f, a, b):
        nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        xm = 0.5 * (a + b)
        xr = 0.5 * (b - a)
        x_map = xm + xr * nodes
        return float(xr * np.sum(weights * f(x_map)))
