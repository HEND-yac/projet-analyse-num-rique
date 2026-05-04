import os
import numpy as np
import matplotlib.pyplot as plt

from polynomial_interpolation import PolynomialInterpolation
from integration import NewtonCotes, AdaptiveIntegration
from applications import CoolingProblem, FlowProblem
from visualization import Visualizer


def load_experimental_data(base_dir):
    cooling_path = os.path.join(base_dir, "data", "cooling_data.csv")
    flow_path = os.path.join(base_dir, "data", "flow_data.csv")

    cooling = np.loadtxt(cooling_path, delimiter=",", skiprows=1)
    flow = np.loadtxt(flow_path, delimiter=",", skiprows=1)

    t_data, T_data = cooling[:, 0], cooling[:, 1]
    x_data, v_data = flow[:, 0], flow[:, 1]
    return t_data, T_data, x_data, v_data


# 🔷 Runge (sans spline)
def run_runge_study(viz, output_dir):
    f_true = lambda x: 1.0 / (1.0 + 25.0 * x**2)
    x_fine = np.linspace(-1.0, 1.0, 500)
    y_true = f_true(x_fine)

    interpolations = {}

    for n_nodes in [5, 11, 17]:
        x_nodes = np.linspace(-1.0, 1.0, n_nodes)
        y_nodes = f_true(x_nodes)

        poly = PolynomialInterpolation(x_nodes, y_nodes)
        y_poly = poly.evaluate(x_fine, method="newton")

        interpolations[f"Polynome Newton (n={n_nodes})"] = y_poly

    fig, _ = viz.plot_runge_phenomenon(x_fine, y_true, interpolations)
    fig.savefig(os.path.join(output_dir, "runge_phenomenon.png"), dpi=160)
    plt.close(fig)

    # Comparaison demandee: une courbe par degre n avec noeuds equidistants vs Tchebychev.
    for n in [5, 10, 15, 20]:
        m = n + 1

        x_eq = np.linspace(-1.0, 1.0, m)
        y_eq_nodes = f_true(x_eq)
        poly_eq = PolynomialInterpolation(x_eq, y_eq_nodes)
        y_eq = poly_eq.evaluate(x_fine, method="newton")

        k = np.arange(m)
        x_cheb = np.cos((2 * k + 1) * np.pi / (2 * m))
        y_cheb_nodes = f_true(x_cheb)
        poly_cheb = PolynomialInterpolation(x_cheb, y_cheb_nodes)
        y_cheb = poly_cheb.evaluate(x_fine, method="newton")

        fig_n, _ = viz.plot_runge_nodes_comparison(x_fine, y_true, y_eq, y_cheb, n)
        fig_n.savefig(os.path.join(output_dir, f"runge_nodes_comparison_n{n}.png"), dpi=160)
        plt.close(fig_n)


# 🔷 Cooling
def run_cooling_analysis(viz, t_data, T_data, output_dir):
    cooling = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)
    t_fine = np.linspace(t_data[0], t_data[-1], 400)

    k_opt = cooling.estimate_k(k_min=0.005, k_max=0.6, tol=1e-5)
    T_model = cooling.exponential_model(t_fine, k_opt)

    total_heat = cooling.total_heat_loss(method="adaptive", n=100)
    model_at_data = cooling.exponential_model(t_data, k_opt)
    abs_errors = np.abs(model_at_data - T_data)
    max_error = float(np.max(abs_errors))
    mean_error = float(np.mean(abs_errors))

    fig, _ = viz.plot_cooling_analysis(t_data, T_data, t_fine, k_opt, T_model)
    fig.savefig(os.path.join(output_dir, "cooling_analysis.png"), dpi=160)
    plt.close(fig)

    # Figure additionnelle: comparaison des interpolations Newton et Lagrange.
    poly = PolynomialInterpolation(t_data, T_data)
    T_newton = poly.evaluate(t_fine, method="newton")
    T_lagrange = poly.evaluate(t_fine, method="lagrange")

    fig_interp, _ = viz.plot_cooling_interpolation_comparison(
        t_data,
        T_data,
        t_fine,
        T_newton,
        T_lagrange,
    )
    fig_interp.savefig(os.path.join(output_dir, "cooling_interpolation_comparison.png"), dpi=160)
    plt.close(fig_interp)

    return {
        "k_opt": k_opt,
        "total_heat_loss": total_heat,
        "model_mse": cooling.model_error(k_opt),
        "model_max_error": max_error,
        "model_mean_error": mean_error,
    }


# 🔷 Flow (corrigé sans spline avancée)
def run_flow_analysis(viz, x_data, v_data, output_dir):
    width_function = lambda x: 0.5 + 0.1 * np.asarray(x)
    flow = FlowProblem(x_data, v_data, width_func=width_function)

    x_fine = np.linspace(x_data[0], x_data[-1], 400)
    v_interp = flow.velocity(x_fine)

    total_q = flow.total_flow_rate(method="adaptive", n=100)

    fig, _ = viz.plot_flow_analysis(x_data, v_data, x_fine, v_interp, width_function)
    fig.savefig(os.path.join(output_dir, "flow_analysis.png"), dpi=160)
    plt.close(fig)

    return {
        "total_flow_rate": total_q
    }


# 🔷 Comparaison méthodes d'intégration – Refroidissement
def run_cooling_integration_comparison(viz, t_data, T_data, output_dir):
    cooling = CoolingProblem(t_data, T_data, T_ambient=20.0, h_coeff=50.0)
    a, b = t_data[0], t_data[-1]
    f = lambda t: cooling.heat_loss_rate(t)
    n = 100

    t_fine = np.linspace(a, b, 500)
    integrand = f(t_fine)

    methods = ["Rectangle", "Trapezes", "Simpson", "Adaptative"]
    q_values = [
        NewtonCotes.rectangle(f, a, b, n),
        NewtonCotes.trapezoidal(f, a, b, n),
        NewtonCotes.simpson(f, a, b, n),
        AdaptiveIntegration(tol=1e-6, max_depth=20).adaptive_simpson(f, a, b),
    ]

    fig, _ = viz.plot_integration_comparison(
        t_fine, integrand, methods, q_values,
        title="Refroidissement : comparaison des méthodes d'intégration"
    )
    fig.savefig(os.path.join(output_dir, "cooling_integration_comparison.png"), dpi=160)
    plt.close(fig)


# 🔷 Comparaison méthodes d'intégration – Écoulement
def run_flow_integration_comparison(viz, x_data, v_data, output_dir):
    width_function = lambda x: 0.5 + 0.1 * np.asarray(x)
    flow = FlowProblem(x_data, v_data, width_func=width_function)
    a, b = float(x_data[0]), float(x_data[-1])
    g = lambda x: flow.local_flow_rate(x)
    n = 100

    x_fine = np.linspace(a, b, 500)
    integrand = g(x_fine)

    methods = ["Rectangle", "Trapezes", "Simpson", "Adaptative"]
    q_values = [
        NewtonCotes.rectangle(g, a, b, n),
        NewtonCotes.trapezoidal(g, a, b, n),
        NewtonCotes.simpson(g, a, b, n),
        AdaptiveIntegration(tol=1e-6, max_depth=20).adaptive_simpson(g, a, b),
    ]

    fig, (ax1, ax2) = viz.plot_integration_comparison(
        x_fine, integrand, methods, q_values,
        title="Écoulement : comparaison des méthodes d'intégration"
    )
    ax1.set_title("Intégrand $v(x) \\cdot w(x)$")
    ax1.set_xlabel("Position x (m)")
    ax1.set_ylabel("Débit local (m²/s)")
    ax2.set_title("Débit total D par méthode (n=100)")
    ax2.set_ylabel("D (m³/s)")
    fig.savefig(os.path.join(output_dir, "flow_integration_comparison.png"), dpi=160)
    plt.close(fig)


# 🔷 Convergence
def run_integration_convergence(viz, output_dir):
    f = np.sin
    a, b = 0.0, np.pi
    exact = 2.0

    n_values = np.array([4, 8, 16, 32, 64, 128])
    methods = ["Rectangle", "Trapezoidal", "Simpson", "Adaptive"]
    errors = {name: [] for name in methods}

    adaptive = AdaptiveIntegration(tol=1e-10, max_depth=30)
    adaptive_value = adaptive.adaptive_simpson(f, a, b)

    for n in n_values:
        rect_val = NewtonCotes.rectangle(f, a, b, n=int(n))
        trap_val = NewtonCotes.trapezoidal(f, a, b, n=int(n))
        simp_val = NewtonCotes.simpson(f, a, b, n=int(n if n % 2 == 0 else n + 1))

        errors["Rectangle"].append(abs(rect_val - exact))
        errors["Trapezoidal"].append(abs(trap_val - exact))
        errors["Simpson"].append(abs(simp_val - exact))
        errors["Adaptive"].append(abs(adaptive_value - exact))

    for k in errors:
        errors[k] = np.asarray(errors[k], dtype=float)

    fig, _ = viz.plot_convergence(
        n_values,
        errors,
        methods,
        title="Convergence des methodes d'integration sur sin(x)",
    )
    fig.savefig(os.path.join(output_dir, "integration_convergence.png"), dpi=160)
    plt.close(fig)

    return {
        "n_values": n_values,
        "errors": errors,
    }


def run_exp_integration_convergence(viz, output_dir):
    f = np.exp
    a, b = 0.0, 1.0
    exact = np.e - 1.0

    n_values = np.array([2, 4, 8, 16, 32, 64, 128, 256])
    methods = ["Rectangle", "Trapezoidal", "Simpson"]
    errors = {name: [] for name in methods}

    for n in n_values:
        rect_val = NewtonCotes.rectangle(f, a, b, n=int(n))
        trap_val = NewtonCotes.trapezoidal(f, a, b, n=int(n))
        simp_val = NewtonCotes.simpson(f, a, b, n=int(n if n % 2 == 0 else n + 1))

        errors["Rectangle"].append(abs(rect_val - exact))
        errors["Trapezoidal"].append(abs(trap_val - exact))
        errors["Simpson"].append(abs(simp_val - exact))

    for k in errors:
        errors[k] = np.asarray(errors[k], dtype=float)

    fig, _ = viz.plot_convergence(
        n_values,
        errors,
        methods,
        title="Convergence des methodes d'integration sur exp(x)",
    )
    fig.savefig(os.path.join(output_dir, "integration_exp_convergence.png"), dpi=160)
    plt.close(fig)

    return {
        "n_values": n_values,
        "errors": errors,
    }


# 🔷 Résumé (corrigé)
def save_summary(output_dir, cooling_results, flow_results, conv_results, exp_conv_results):
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Resultats numeriques ===\n\n")

        f.write("[Refroidissement]\n")
        f.write(f"k optimal              : {cooling_results['k_opt']:.6f}\n")
        f.write(f"Perte totale de chaleur: {cooling_results['total_heat_loss']:.6f}\n")
        f.write(f"Erreur MSE du modele   : {cooling_results['model_mse']:.6f}\n\n")
        f.write(f"Erreur maximale        : {cooling_results['model_max_error']:.6f}\n")
        f.write(f"Erreur moyenne         : {cooling_results['model_mean_error']:.6f}\n\n")

        f.write("[Ecoulement]\n")
        f.write(f"Debit total            : {flow_results['total_flow_rate']:.6f}\n\n")

        f.write("[Convergence integration]\n")
        methods = ["Rectangle", "Trapezoidal", "Simpson", "Adaptive"]
        for i, n in enumerate(conv_results["n_values"]):
            f.write(f"n={int(n):3d}")
            for method in methods:
                val = conv_results["errors"][method][i]
                f.write(f"  {method}:{val:.3e}")
            f.write("\n")

        f.write("\n[Convergence integration exp(x) sur [0,1]]\n")
        exp_methods = ["Rectangle", "Trapezoidal", "Simpson"]
        for i, n in enumerate(exp_conv_results["n_values"]):
            f.write(f"n={int(n):3d}")
            for method in exp_methods:
                val = exp_conv_results["errors"][method][i]
                f.write(f"  {method}:{val:.3e}")
            f.write("\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "results")
    os.makedirs(output_dir, exist_ok=True)

    t_data, T_data, x_data, v_data = load_experimental_data(base_dir)
    viz = Visualizer(style="seaborn-v0_8-darkgrid", figsize=(10, 6))

    run_runge_study(viz, output_dir)
    cooling_results = run_cooling_analysis(viz, t_data, T_data, output_dir)
    run_cooling_integration_comparison(viz, t_data, T_data, output_dir)
    flow_results = run_flow_analysis(viz, x_data, v_data, output_dir)
    run_flow_integration_comparison(viz, x_data, v_data, output_dir)
    conv_results = run_integration_convergence(viz, output_dir)
    exp_conv_results = run_exp_integration_convergence(viz, output_dir)

    save_summary(output_dir, cooling_results, flow_results, conv_results, exp_conv_results)

    print("✅ Execution terminee. Resultats dans le dossier 'results'.")


if __name__ == "__main__":
    main()