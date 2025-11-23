"""
Markov Chain Implementation
Unidad 3: Cadenas de Markov

Una cadena de Markov es un sistema estocástico que transita
entre estados con probabilidades definidas.

P(X_{t+1} = j | X_t = i) = P_{ij}

Aplicación: Modelar el flujo de trabajo de documentos
"""

import numpy as np
from collections import defaultdict
import json


class MarkovChain:
    """
    Implementación de una Cadena de Markov de Tiempo Discreto (DTMC).

    Una cadena de Markov se define por:
    - Conjunto de estados S = {s1, s2, ..., sn}
    - Matriz de transición P donde P[i][j] = P(ir de i a j)
    - Distribución inicial π

    Propiedades de la matriz de transición:
    - Cada fila suma 1 (matriz estocástica)
    - P[i][j] ≥ 0 para todo i, j

    Atributos:
    ----------
    states : list
        Lista de nombres de estados
    transition_matrix : np.array
        Matriz de transición nxn
    state_to_index : dict
        Mapeo estado → índice
    """

    def __init__(self, states):
        """
        Inicializa la cadena de Markov.

        Parámetros:
        -----------
        states : list
            Lista de nombres de estados

        Ejemplo:
        --------
        >>> states = ['received', 'classified', 'processed']
        >>> mc = MarkovChain(states)
        """
        self.states = states
        self.num_states = len(states)

        # Crear mapeo estado <-> índice
        self.state_to_index = {state: i for i, state in enumerate(states)}
        self.index_to_state = {i: state for i, state in enumerate(states)}

        # Inicializar matriz de transición (uniforme por defecto)
        self.transition_matrix = np.ones((self.num_states, self.num_states))
        self.transition_matrix /= self.num_states

        # Distribución inicial (uniforme por defecto)
        self.initial_distribution = np.ones(self.num_states) / self.num_states

    def set_transition_probability(self, from_state, to_state, probability):
        """
        Establece la probabilidad de transición entre dos estados.

        Parámetros:
        -----------
        from_state : str
            Estado origen
        to_state : str
            Estado destino
        probability : float
            Probabilidad de transición [0, 1]

        Ejemplo:
        --------
        >>> mc.set_transition_probability('received', 'classified', 0.8)
        """
        i = self.state_to_index[from_state]
        j = self.state_to_index[to_state]
        self.transition_matrix[i, j] = probability

    def set_transition_matrix(self, matrix):
        """
        Establece toda la matriz de transición.

        Parámetros:
        -----------
        matrix : np.array or list
            Matriz de transición nxn

        Ejemplo:
        --------
        >>> matrix = [[0.7, 0.3],
        ...           [0.4, 0.6]]
        >>> mc.set_transition_matrix(matrix)
        """
        self.transition_matrix = np.array(matrix, dtype=float)

        # Validar que sea estocástica
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            print("Advertencia: Las filas no suman 1. Normalizando...")
            self.transition_matrix = self.transition_matrix / row_sums[:, np.newaxis]

    def set_initial_distribution(self, distribution):
        """
        Establece la distribución inicial.

        Parámetros:
        -----------
        distribution : list or np.array
            Distribución de probabilidad inicial
        """
        self.initial_distribution = np.array(distribution, dtype=float)

        # Normalizar si es necesario
        total = self.initial_distribution.sum()
        if not np.isclose(total, 1.0):
            self.initial_distribution /= total

    def get_transition_probability(self, from_state, to_state):
        """
        Obtiene la probabilidad de transición entre dos estados.

        Parámetros:
        -----------
        from_state : str
            Estado origen
        to_state : str
            Estado destino

        Retorna:
        --------
        float
            Probabilidad de transición
        """
        i = self.state_to_index[from_state]
        j = self.state_to_index[to_state]
        return self.transition_matrix[i, j]

    def next_state(self, current_state):
        """
        Simula la transición al siguiente estado.

        Usa la distribución de probabilidad de la fila correspondiente
        al estado actual para elegir el siguiente estado.

        Parámetros:
        -----------
        current_state : str
            Estado actual

        Retorna:
        --------
        str
            Siguiente estado

        Ejemplo:
        --------
        >>> current = 'received'
        >>> next_state = mc.next_state(current)
        >>> print(next_state)
        'classified'
        """
        i = self.state_to_index[current_state]

        # Obtener probabilidades de transición
        probabilities = self.transition_matrix[i, :]

        # Elegir siguiente estado según probabilidades
        next_index = np.random.choice(self.num_states, p=probabilities)

        return self.index_to_state[next_index]

    def simulate(self, num_steps, initial_state=None):
        """
        Simula una trayectoria de la cadena de Markov.

        Parámetros:
        -----------
        num_steps : int
            Número de pasos a simular
        initial_state : str, optional
            Estado inicial (aleatorio si no se especifica)

        Retorna:
        --------
        list
            Lista de estados visitados

        Ejemplo:
        --------
        >>> trajectory = mc.simulate(100, initial_state='received')
        >>> print(trajectory[:5])
        ['received', 'classified', 'processed', 'processed', 'archived']
        """
        # Elegir estado inicial
        if initial_state is None:
            current_state = np.random.choice(self.states, p=self.initial_distribution)
        else:
            current_state = initial_state

        trajectory = [current_state]

        # Simular transiciones
        for _ in range(num_steps - 1):
            current_state = self.next_state(current_state)
            trajectory.append(current_state)

        return trajectory

    def stationary_distribution(self, method="eigenvalue", max_iterations=1000):
        """
        Calcula la distribución estacionaria.

        La distribución estacionaria π satisface: π * P = π

        Dos métodos:
        1. Eigenvalue: Encuentra el eigenvector del eigenvalor 1
        2. Power iteration: Multiplica P por sí misma hasta convergencia

        Parámetros:
        -----------
        method : str
            'eigenvalue' o 'power'
        max_iterations : int
            Máximo de iteraciones (solo para power)

        Retorna:
        --------
        np.array
            Distribución estacionaria

        Ejemplo:
        --------
        >>> pi = mc.stationary_distribution()
        >>> print(pi)
        [0.3, 0.2, 0.25, 0.15, 0.1]
        """
        if method == "eigenvalue":
            # Encontrar eigenvalores y eigenvectores
            eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)

            # Encontrar eigenvector correspondiente a eigenvalor ≈ 1
            idx = np.argmax(np.abs(eigenvalues - 1.0) < 1e-10)

            # Obtener eigenvector y normalizar
            stationary = np.real(eigenvectors[:, idx])
            stationary = stationary / stationary.sum()

            return stationary

        elif method == "power":
            # Método de iteración de potencia
            pi = self.initial_distribution.copy()

            for _ in range(max_iterations):
                pi_new = pi @ self.transition_matrix

                # Verificar convergencia
                if np.allclose(pi_new, pi, atol=1e-8):
                    return pi_new

                pi = pi_new

            print("Advertencia: No convergió en", max_iterations, "iteraciones")
            return pi

        else:
            raise ValueError(f"Método desconocido: {method}")

    def n_step_transition(self, n):
        """
        Calcula la matriz de transición de n pasos.

        P^(n) = P * P * ... * P (n veces)

        P^(n)[i][j] = probabilidad de ir de i a j en exactamente n pasos

        Parámetros:
        -----------
        n : int
            Número de pasos

        Retorna:
        --------
        np.array
            Matriz de transición de n pasos
        """
        return np.linalg.matrix_power(self.transition_matrix, n)

    def expected_time_to_state(self, target_state, from_state=None):
        """
        Calcula el tiempo esperado para llegar a un estado.

        (Hitting time / First passage time)

        Parámetros:
        -----------
        target_state : str
            Estado objetivo
        from_state : str, optional
            Estado inicial (promedio sobre todos si no se especifica)

        Retorna:
        --------
        float
            Tiempo esperado (número de pasos)
        """
        # Este es un cálculo simplificado
        # Para el cálculo exacto se resuelve un sistema lineal

        target_idx = self.state_to_index[target_state]

        if from_state is None:
            # Simular muchas veces y promediar
            times = []
            for _ in range(1000):
                initial = np.random.choice(self.states)
                trajectory = self.simulate(1000, initial)

                try:
                    time = trajectory.index(target_state)
                    times.append(time)
                except ValueError:
                    continue

            return np.mean(times) if times else float("inf")
        else:
            # Simular desde estado específico
            times = []
            for _ in range(1000):
                trajectory = self.simulate(1000, from_state)

                try:
                    time = trajectory.index(target_state)
                    times.append(time)
                except ValueError:
                    continue

            return np.mean(times) if times else float("inf")

    def visualize_transition_matrix(self):
        """
        Visualiza la matriz de transición.

        Muestra la matriz con nombres de estados.
        """
        print("\n" + "=" * 60)
        print("MATRIZ DE TRANSICIÓN")
        print("=" * 60)

        # Encabezado
        header = "De \\ A".ljust(15)
        for state in self.states:
            header += f"{state[:10]:>12}"
        print(header)
        print("-" * 60)

        # Filas
        for i, from_state in enumerate(self.states):
            row = from_state[:12].ljust(15)
            for j in range(self.num_states):
                prob = self.transition_matrix[i, j]
                row += f"{prob:12.4f}"
            print(row)

        print("=" * 60)

    def get_statistics(self):
        """
        Obtiene estadísticas de la cadena.

        Retorna:
        --------
        dict
            Estadísticas
        """
        stationary = self.stationary_distribution()

        return {
            "num_states": self.num_states,
            "states": self.states,
            "stationary_distribution": dict(zip(self.states, stationary)),
            "is_irreducible": self._is_irreducible(),
            "is_aperiodic": self._is_aperiodic(),
        }

    def _is_irreducible(self):
        """
        Verifica si la cadena es irreducible.

        Una cadena es irreducible si se puede llegar de cualquier
        estado a cualquier otro estado.
        """
        # Calcular P^n para n grande
        P_large = self.n_step_transition(self.num_states * 10)

        # Si todos los elementos son > 0, es irreducible
        return np.all(P_large > 0)

    def _is_aperiodic(self):
        """
        Verifica si la cadena es aperiódica.

        Una cadena es aperiódica si el período de cada estado es 1.
        """
        # Simplificación: verificar si hay auto-loops
        return np.any(np.diag(self.transition_matrix) > 0)

    def __str__(self):
        """Representación en string"""
        return f"MarkovChain(states={self.num_states})"


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("CADENA DE MARKOV - DEMOSTRACIÓN")
    print("=" * 60)

    # Ejemplo 1: Flujo de documentos
    print("\n--- EJEMPLO 1: FLUJO DE DOCUMENTOS ---")

    # Estados del documento
    states = ["received", "classified", "processed", "archived", "retrieved"]

    mc = MarkovChain(states)

    # Definir matriz de transición
    # Filas: estado actual, Columnas: siguiente estado
    transition = [
        # to: recv  class  proc  arch  retr
        [0.10, 0.70, 0.10, 0.05, 0.05],  # from: received
        [0.05, 0.10, 0.70, 0.10, 0.05],  # from: classified
        [0.00, 0.05, 0.20, 0.70, 0.05],  # from: processed
        [0.00, 0.00, 0.10, 0.70, 0.20],  # from: archived
        [0.00, 0.00, 0.05, 0.80, 0.15],  # from: retrieved
    ]

    mc.set_transition_matrix(transition)

    # Visualizar matriz
    mc.visualize_transition_matrix()

    # Calcular distribución estacionaria
    print("\n--- DISTRIBUCIÓN ESTACIONARIA ---")
    stationary = mc.stationary_distribution()

    print("\nProbabilidad a largo plazo de estar en cada estado:")
    for state, prob in zip(states, stationary):
        print(f"  {state:12}: {prob:.4f} ({prob*100:.2f}%)")

    # Simular trayectoria
    print("\n--- SIMULACIÓN ---")

    trajectory = mc.simulate(50, initial_state="received")

    print(f"\nTrayectoria de 50 pasos:")
    print(" -> ".join(trajectory[:10]) + " -> ...")

    # Contar estados visitados
    from collections import Counter

    state_counts = Counter(trajectory)

    print("\nDistribución observada:")
    for state in states:
        count = state_counts[state]
        freq = count / len(trajectory)
        print(f"  {state:12}: {count:2d}/50 = {freq:.4f}")

    # Tiempo esperado
    print("\n--- TIEMPOS ESPERADOS ---")

    for target in ["archived", "retrieved"]:
        time = mc.expected_time_to_state(target, "received")
        print(f"  De 'received' a '{target}': ~{time:.1f} pasos")

    # Estadísticas
    print("\n--- ESTADÍSTICAS DE LA CADENA ---")
    stats = mc.get_statistics()

    print(f"  Número de estados: {stats['num_states']}")
    print(f"  Irreducible: {stats['is_irreducible']}")
    print(f"  Aperiódica: {stats['is_aperiodic']}")

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)
