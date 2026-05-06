import numpy as np
from integration import NewtonCotes, AdaptiveIntegration

class CoolingProblem:
    def __init__(self, t_data, T_data, T_ambient=20.0, h_coeff=50.0):
        self.t_data = np.asarray(t_data, dtype=float)
        self.T_data = np.asarray(T_data, dtype=float)
        self.T_ambient = float(T_ambient)
        self.h_coeff = float(h_coeff)

        if self.t_data.ndim != 1 or self.T_data.ndim != 1:
            raise ValueError("t_data et T_data doivent etre 1D")
        if self.t_data.size != self.T_data.size:
            raise ValueError("tailles différentes")
        if np.any(np.diff(self.t_data) <= 0):
            raise ValueError("t_data doit etre croissant")

        # Pre-calculs interpolation
        self.newton_coeffs = self.newton_coefficients()

    # 🔷 NEWTON COEFFS
    def newton_coefficients(self):
        x = self.t_data
        y = self.T_data.copy()
        n = len(x)

        for j in range(1, n):
            y[j:n] = (y[j:n] - y[j-1:n-1]) / (x[j:n] - x[0:n-j])

        return y

    # 🔷 EVALUATION NEWTON
    def temperature(self, t_eval):
        x = self.t_data
        coeffs = self.newton_coeffs

        t = np.atleast_1d(t_eval)
        values = np.full_like(t, coeffs[-1], dtype=float)

        for i in range(len(coeffs)-2, -1, -1):
            values = values * (t - x[i]) + coeffs[i]

        if np.isscalar(t_eval):
            return float(values[0])
        return values



    def heat_loss_rate(self, t_eval):
        T = self.temperature(t_eval)
        return self.h_coeff * (T - self.T_ambient)

    def total_heat_loss(self, method="adaptive", n=100):
        a, b = self.t_data[0], self.t_data[-1]
        f = lambda t: self.heat_loss_rate(t)

        if method == "adaptive":
            integ = AdaptiveIntegration()
            return integ.adaptive_simpson(f, a, b)

        if method == "trapezoidal":
            return NewtonCotes.trapezoidal(f, a, b, n)

        if method == "rectangle":
            return NewtonCotes.rectangle(f, a, b, n)

        if method == "simpson":
            if n % 2 != 0:
                n += 1
            return NewtonCotes.simpson(f, a, b, n)

        raise ValueError("methode inconnue")
    def exponential_model(self, t_eval, k):
        t = np.asarray(t_eval, dtype=float)
        t0 = self.t_data[0]
        T0 = self.T_data[0]

        values = self.T_ambient + (T0 - self.T_ambient) * np.exp(-k * (t - t0))

        if np.isscalar(t_eval):
            return float(values[0])
        return values
 

    def model_error(self, k):
        model_vals = self.exponential_model(self.t_data, k)
        return np.mean((model_vals - self.T_data) ** 2)


    def estimate_k(self, k_min=0.01, k_max=0.5, tol=1e-4):
        a, b = k_min, k_max
        phi = (1 + np.sqrt(5)) / 2

        c = b - (b - a) / phi
        d = a + (b - a) / phi

        while abs(b - a) > tol:
            if self.model_error(c) < self.model_error(d):
                b = d
            else:
                a = c

            c = b - (b - a) / phi
            d = a + (b - a) / phi

        return (a + b) / 2


# 🔷 FLOW PROBLEM (SANS SPLINE)
class FlowProblem:
    def __init__(self, x_data, v_data, width_func=None):
        self.x_data = np.asarray(x_data, dtype=float)
        self.v_data = np.asarray(v_data, dtype=float)

        if self.x_data.ndim != 1 or self.v_data.ndim != 1:
            raise ValueError("x_data et v_data doivent etre 1D")

        self.width_func = width_func if width_func else (lambda x: 0.5 + 0.1 * x)

        self.coeffs = self.newton_coefficients()

    def newton_coefficients(self):
        x = self.x_data
        y = self.v_data.copy()
        n = len(x)

        for j in range(1, n):
            y[j:n] = (y[j:n] - y[j-1:n-1]) / (x[j:n] - x[0:n-j])

        return y

    def velocity(self, x_eval):
        x = self.x_data
        coeffs = self.coeffs

        t = np.atleast_1d(x_eval)
        values = np.full_like(t, coeffs[-1], dtype=float)

        for i in range(len(coeffs)-2, -1, -1):
            values = values * (t - x[i]) + coeffs[i]

        if np.isscalar(x_eval):
            return float(values[0])
        return values

    def local_flow_rate(self, x_eval):
        return self.velocity(x_eval) * self.width_func(x_eval)

    def total_flow_rate(self, method="adaptive", n=100):
        a, b = self.x_data[0], self.x_data[-1]
        f = lambda x: self.local_flow_rate(x)

        if method == "adaptive":
            integ = AdaptiveIntegration()
            return integ.adaptive_simpson(f, a, b)

        if method == "trapezoidal":
            return NewtonCotes.trapezoidal(f, a, b, n)

        return NewtonCotes.rectangle(f, a, b, n)