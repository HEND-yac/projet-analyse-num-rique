import matplotlib.pyplot as plt


class Visualizer:
    """Utilitaires de tracage matplotlib pour les analyses du projet."""

    def __init__(self, style="seaborn-v0_8-darkgrid", figsize=(10, 6)):
        self.style = style
        self.figsize = figsize
        plt.style.use(self.style)

    def plot_interpolation_comparison(self, x_data, y_data, interpolators, x_fine, title):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(x_data, y_data, c="black", s=40, label="Donnees")

        for name, values in interpolators.items():
            ax.plot(x_fine, values, linewidth=2.0, label=name)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_runge_phenomenon(self, x_fine, y_true, interpolations):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x_fine, y_true, "k-", linewidth=2.5, label="f(x) vraie")

        for name, y_interp in interpolations.items():
            ax.plot(x_fine, y_interp, linewidth=1.8, label=name)

        ax.set_title("Phenomene de Runge")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_runge_nodes_comparison(self, x_fine, y_true, y_eq, y_cheb, n):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(x_fine, y_true, "k-", linewidth=2.5, label="f(x) vraie")
        ax.plot(x_fine, y_eq, linewidth=2.0, label=f"Equidistants (n={n})")
        ax.plot(x_fine, y_cheb, "--", linewidth=2.0, label=f"Tchebychev (n={n})")

        ax.set_title(f"Runge: comparaison des noeuds pour n={n}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_convergence(self, n_values, errors, methods, title):
        fig, ax = plt.subplots(figsize=self.figsize)
        for method in methods:
            ax.loglog(n_values, errors[method], marker="o", linewidth=2.0, label=method)

        ax.set_title(title)
        ax.set_xlabel("Nombre de sous-intervalles n")
        ax.set_ylabel("Erreur absolue")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_cooling_analysis(self, t_data, T_data, t_fine, k_opt, T_model):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(t_data, T_data, c="black", s=40, label="Mesures")
        ax.plot(t_fine, T_model, "--", linewidth=2.0, label=f"Modele exponentiel (k={k_opt:.4f})")

        ax.set_title("Analyse du refroidissement")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Temperature")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_cooling_interpolation_comparison(self, t_data, T_data, t_fine, T_newton, T_lagrange):
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.scatter(t_data, T_data, c="black", s=40, label="Mesures")
        ax.plot(t_fine, T_newton, linewidth=2.2, label="Interpolation Newton")
        ax.plot(t_fine, T_lagrange, "--", linewidth=2.0, label="Interpolation Lagrange")

        ax.set_title("Refroidissement: comparaison Newton vs Lagrange")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Temperature")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_integration_comparison(self, t_fine, integrand, methods, q_values, title="Comparaison des methodes d'integration"):
        """Deux sous-figures: (1) intégrand + aire, (2) Q par méthode."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # --- gauche : courbe de l'intégrand ---
        ax1.fill_between(t_fine, integrand, alpha=0.25, label="Aire (Q)")
        ax1.plot(t_fine, integrand, linewidth=2.5, color="tab:blue")
        ax1.set_title("Intégrand $h\\cdot(T(t)-T_{amb})$")
        ax1.set_xlabel("Temps t (s)")
        ax1.set_ylabel("Flux de chaleur (J/s)")
        ax1.legend()

        # --- droite : valeurs de Q par méthode ---
        colors = ["tab:orange", "tab:green", "tab:red", "tab:purple"]
        bars = ax2.bar(methods, q_values, color=colors[:len(methods)], width=0.5, edgecolor="black")
        ax2.set_title("Chaleur dissipée Q par méthode (n=100)")
        ax2.set_xlabel("Méthode")
        ax2.set_ylabel("Q (J)")
        # annoter chaque barre avec la valeur
        for bar, val in zip(bars, q_values):
            ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        # zoom sur l'écart réel
        margin = max(abs(v - min(q_values)) for v in q_values) * 3
        ax2.set_ylim(min(q_values) - margin, max(q_values) + margin * 2)

        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        return fig, (ax1, ax2)

    def plot_flow_analysis(self, x_data, v_data, x_fine, v_interp, w_function):
        fig, ax1 = plt.subplots(figsize=self.figsize)
        ax1.scatter(x_data, v_data, c="black", s=40, label="Vitesse mesuree")
        ax1.plot(x_fine, v_interp, linewidth=2.2, color="tab:blue", label="Vitesse interpolee")
        ax1.set_xlabel("Position x")
        ax1.set_ylabel("Vitesse", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(x_fine, w_function(x_fine), "--", linewidth=2.0, color="tab:red", label="Largeur")
        ax2.set_ylabel("Largeur", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

        ax1.set_title("Analyse de l'ecoulement")
        fig.tight_layout()
        return fig, (ax1, ax2)
