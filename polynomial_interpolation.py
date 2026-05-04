import numpy as np
import matplotlib.pyplot as plt


class PolynomialInterpolation:
    """Interpolation polynomiale via les formulations de Lagrange et Newton."""

    def __init__(self, x_points, y_points):
        """
        Initialise l'interpolateur polynômial.

        Parameters
        ----------
        x_points : array-like
            Abscisses des points d'interpolation.
        y_points : array-like
            Ordonnées des points d'interpolation.

        Raises
        ------
        ValueError
            Si les longueurs de x_points et y_points diffèrent,
            si aucune donnée n'est fournie, ou si des abscisses sont non distinctes.
        """
        self.x_points = np.asarray(x_points, dtype=float)
        self.y_points = np.asarray(y_points, dtype=float)

        if self.x_points.ndim != 1 or self.y_points.ndim != 1:
            raise ValueError("x_points et y_points doivent être des tableaux 1D.")

        if self.x_points.size == 0:
            raise ValueError("Au moins un point d'interpolation est requis.")

        if self.x_points.size != self.y_points.size:
            raise ValueError("x_points et y_points doivent avoir la même longueur.")

        if np.unique(self.x_points).size != self.x_points.size:
            raise ValueError("Les abscisses x_points doivent être toutes distinctes.")

        self._newton_coeffs = None

    def lagrange(self, x_eval):
        """
        Évalue le polynôme interpolateur sous forme de Lagrange.

        Parameters
        ----------
        x_eval : float or array-like
            Point(s) où évaluer le polynôme.

        Returns
        -------
        float or numpy.ndarray
            Valeur(s) interpolée(s). Retourne un scalaire si x_eval est scalaire.
        """
        x = self.x_points
        y = self.y_points

        x_eval_arr = np.asarray(x_eval, dtype=float)
        is_scalar = x_eval_arr.ndim == 0
        x_eval_1d = np.atleast_1d(x_eval_arr)

        # Matrice des différences x_i - x_j pour construire les dénominateurs L_i.
        diff_x = x[:, None] - x[None, :]
        np.fill_diagonal(diff_x, 1.0)
        denom = np.prod(diff_x, axis=1)

        # Matrice (x_eval_k - x_i) de taille (m, n).
        eval_diff = x_eval_1d[:, None] - x[None, :]

        # Calcul stable des bases de Lagrange via produit total / terme exclu.
        # Division protegee pour eviter les avertissements sur les coincidences exactes.
        total_prod = np.prod(eval_diff, axis=1)
        basis = np.divide(
            total_prod[:, None],
            eval_diff,
            out=np.zeros_like(eval_diff),
            where=~np.isclose(eval_diff, 0.0),
        )
        values = basis @ (y / denom)

        # Corrige les points d'évaluation coïncidant exactement avec des x_i.
        exact_matches = np.isclose(eval_diff, 0.0)
        if np.any(exact_matches):
            rows, cols = np.where(exact_matches)
            values[rows] = y[cols]

        if is_scalar:
            return float(values[0])
        return values

    def newton_coefficients(self):
        """
        Calcule les coefficients de Newton par différences divisées.

        Returns
        -------
        numpy.ndarray
            Coefficients de Newton c_0, c_1, ..., c_{n-1}.
        """
        if self._newton_coeffs is not None:
            return self._newton_coeffs.copy()

        x = self.x_points
        coeffs = self.y_points.astype(float).copy()
        n = x.size

        for j in range(1, n):
            coeffs[j:n] = (coeffs[j:n] - coeffs[j - 1 : n - 1]) / (x[j:n] - x[0 : n - j])

        self._newton_coeffs = coeffs
        return coeffs.copy()

    def newton_eval(self, x_eval):
        """
        Évalue le polynôme interpolateur sous forme de Newton.

        Parameters
        ----------
        x_eval : float or array-like
            Point(s) où évaluer le polynôme.

        Returns
        -------
        float or numpy.ndarray
            Valeur(s) interpolée(s). Retourne un scalaire si x_eval est scalaire.
        """
        coeffs = self.newton_coefficients()
        x_nodes = self.x_points

        x_eval_arr = np.asarray(x_eval, dtype=float)
        is_scalar = x_eval_arr.ndim == 0
        values = np.full_like(np.atleast_1d(x_eval_arr), coeffs[-1], dtype=float)

        for k in range(coeffs.size - 2, -1, -1):
            values = values * (np.atleast_1d(x_eval_arr) - x_nodes[k]) + coeffs[k]

        if is_scalar:
            return float(values[0])
        return values

    def evaluate(self, x_eval, method="newton"):
        """
        Interface unifiée d'évaluation du polynôme interpolateur.

        Parameters
        ----------
        x_eval : float or array-like
            Point(s) où évaluer le polynôme.
        method : str, default='newton'
            Méthode à utiliser: 'newton' ou 'lagrange'.

        Returns
        -------
        float or numpy.ndarray
            Valeur(s) interpolée(s).

        Raises
        ------
        ValueError
            Si la méthode demandée n'est pas reconnue.
        """
        method_normalized = method.lower()

        if method_normalized == "newton":
            return self.newton_eval(x_eval)
        if method_normalized == "lagrange":
            return self.lagrange(x_eval)

        raise ValueError("Méthode invalide. Utilisez 'newton' ou 'lagrange'.")




