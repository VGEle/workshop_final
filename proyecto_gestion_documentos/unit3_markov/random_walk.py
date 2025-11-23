"""
Random Walk Analysis
Unidad 3: Cadenas de Markov - Random Walk

Una caminata aleatoria (random walk) es una trayectoria que consiste
en tomar pasos sucesivos aleatorios en un grafo.

Conceptos clave:
- Cover Time: tiempo para visitar todos los nodos
- Mixing Time: tiempo para alcanzar distribución estacionaria
- Hitting Time: tiempo para llegar a un nodo específico

Aplicación: Analizar cómo usuarios navegan por documentos
"""

import numpy as np
from collections import defaultdict, Counter
import random


class RandomWalk:
    """
    Implementación de Random Walk en grafos.

    Una caminata aleatoria en un grafo:
    1. Comienza en un nodo
    2. En cada paso, elige un vecino aleatoriamente
    3. Se mueve a ese vecino
    4. Repite

    Atributos:
    ----------
    graph : dict
        Grafo como dict: nodo → [vecinos]
    nodes : list
        Lista de nodos
    """

    def __init__(self):
        """
        Inicializa Random Walk.

        Ejemplo:
        --------
        >>> rw = RandomWalk()
        """
        self.graph = defaultdict(list)
        self.nodes = []

    def add_edge(self, from_node, to_node):
        """
        Añade una arista al grafo.

        Parámetros:
        -----------
        from_node : str
            Nodo origen
        to_node : str
            Nodo destino
        """
        if from_node not in self.nodes:
            self.nodes.append(from_node)
        if to_node not in self.nodes:
            self.nodes.append(to_node)

        # Añadir arista (dirigida)
        if to_node not in self.graph[from_node]:
            self.graph[from_node].append(to_node)

    def build_from_dict(self, graph_dict):
        """
        Construye el grafo desde un diccionario.

        Parámetros:
        -----------
        graph_dict : dict
            Diccionario nodo → [vecinos]
        """
        self.graph = defaultdict(list, graph_dict)
        self.nodes = list(graph_dict.keys())

    def walk(self, start_node, num_steps):
        """
        Realiza una caminata aleatoria.

        Parámetros:
        -----------
        start_node : str
            Nodo inicial
        num_steps : int
            Número de pasos

        Retorna:
        --------
        list
            Secuencia de nodos visitados

        Ejemplo:
        --------
        >>> rw = RandomWalk()
        >>> rw.add_edge('A', 'B')
        >>> rw.add_edge('B', 'C')
        >>> rw.add_edge('C', 'A')
        >>> path = rw.walk('A', 10)
        >>> print(path)
        ['A', 'B', 'C', 'A', 'B', 'C', 'C', 'A', 'B', 'C', 'A']
        """
        current = start_node
        path = [current]

        for _ in range(num_steps):
            neighbors = self.graph[current]

            if not neighbors:
                path.append(current)
            else:
                current = random.choice(neighbors)
                path.append(current)

        return path

    def estimate_hitting_time(self, start_node, target_node, num_trials=1000, max_steps=10000):
        """
        Estima el hitting time (tiempo de llegada).

        Hitting time: número esperado de pasos para llegar
        desde start_node a target_node.

        Parámetros:
        -----------
        start_node : str
            Nodo inicial
        target_node : str
            Nodo objetivo
        num_trials : int
            Número de simulaciones
        max_steps : int
            Máximo de pasos por simulación

        Retorna:
        --------
        float
            Hitting time estimado
        """
        times = []

        for _ in range(num_trials):
            path = self.walk(start_node, max_steps)

            try:
                time = path.index(target_node)
                times.append(time)
            except ValueError:
                continue

        if not times:
            return float("inf")

        return np.mean(times)

    def estimate_cover_time(self, start_node, num_trials=100, max_steps=100000):
        """
        Estima el cover time (tiempo de cobertura).

        Cover time: número esperado de pasos para visitar
        todos los nodos al menos una vez.

        Parámetros:
        -----------
        start_node : str
            Nodo inicial
        num_trials : int
            Número de simulaciones
        max_steps : int
            Máximo de pasos

        Retorna:
        --------
        float
            Cover time estimado
        """
        times = []

        for _ in range(num_trials):
            visited = set()
            current = start_node
            visited.add(current)

            steps = 0
            while len(visited) < len(self.nodes) and steps < max_steps:
                neighbors = self.graph[current]

                if neighbors:
                    current = random.choice(neighbors)
                    visited.add(current)

                steps += 1

            if len(visited) == len(self.nodes):
                times.append(steps)

        if not times:
            return float("inf")

        return np.mean(times)

    def estimate_mixing_time(self, num_trials=100, num_steps=1000, tolerance=0.01):
        """
        Estima el mixing time (tiempo de mezcla).

        Mixing time: número de pasos necesarios para que
        la distribución de probabilidad esté cerca de
        la distribución estacionaria.

        Parámetros:
        -----------
        num_trials : int
            Número de simulaciones
        num_steps : int
            Pasos por simulación
        tolerance : float
            Tolerancia para convergencia

        Retorna:
        --------
        int
            Mixing time estimado
        """
        stationary = {node: 1.0 / len(self.nodes) for node in self.nodes}

        mixing_times = []

        for _ in range(num_trials):
            start = random.choice(self.nodes)

            for t in range(1, num_steps):
                path = self.walk(start, t)

                counts = Counter(path)
                distribution = {node: counts[node] / len(path) for node in self.nodes}

                distance = sum(
                    abs(distribution.get(node, 0) - stationary[node]) for node in self.nodes
                )

                if distance < tolerance:
                    mixing_times.append(t)
                    break

        if not mixing_times:
            return num_steps

        return int(np.mean(mixing_times))

    def analyze_walk(self, path):
        """
        Analiza una caminata aleatoria.

        Parámetros:
        -----------
        path : list
            Secuencia de nodos

        Retorna:
        --------
        dict
            Análisis de la caminata
        """
        visit_counts = Counter(path)
        unique_nodes = len(set(path))
        total = len(path)
        frequencies = {node: count / total for node, count in visit_counts.items()}

        return {
            "length": total,
            "unique_nodes": unique_nodes,
            "coverage": unique_nodes / len(self.nodes),
            "visit_counts": dict(visit_counts),
            "frequencies": frequencies,
            "most_visited": visit_counts.most_common(5),
        }

    def simulate_user_navigation(self, start_doc, num_clicks=20, return_probability=0.1):
        """
        Simula navegación de usuario con posibilidad de retorno.

        Modelo más realista donde el usuario puede:
        1. Seguir un link (90%)
        2. Volver al inicio (10%)

        Parámetros:
        -----------
        start_doc : str
            Documento inicial
        num_clicks : int
            Número de clicks
        return_probability : float
            Probabilidad de volver al inicio

        Retorna:
        --------
        list
            Secuencia de documentos visitados
        """
        current = start_doc
        path = [current]

        for _ in range(num_clicks):
            if random.random() < return_probability:
                current = start_doc
            else:
                neighbors = self.graph[current]
                if neighbors:
                    current = random.choice(neighbors)

            path.append(current)

        return path

    def get_statistics(self):
        """
        Obtiene estadísticas del grafo.

        Retorna:
        --------
        dict
            Estadísticas
        """
        out_degrees = {node: len(self.graph[node]) for node in self.nodes}
        in_degrees = defaultdict(int)

        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                in_degrees[neighbor] += 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(neighbors) for neighbors in self.graph.values()),
            "avg_out_degree": np.mean(list(out_degrees.values())) if out_degrees else 0,
            "max_out_degree": max(out_degrees.values()) if out_degrees else 0,
            "min_out_degree": min(out_degrees.values()) if out_degrees else 0,
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("RANDOM WALK - DEMOSTRACIÓN")
    print("=" * 60)

    # Ejemplo 1: Grafo simple
    print("\n--- EJEMPLO 1: GRAFO SIMPLE ---")

    rw = RandomWalk()

    rw.add_edge("A", "B")
    rw.add_edge("B", "C")
    rw.add_edge("C", "A")

    print("Grafo circular: A -> B -> C -> A")

    path = rw.walk("A", 20)
    print(f"\nCaminata de 20 pasos desde A:")
    print(" -> ".join(path[:15]) + " -> ...")

    analysis = rw.analyze_walk(path)
    print(f"\nAnálisis:")
    print(f"  Nodos únicos visitados: {analysis['unique_nodes']}")
    print(f"  Cobertura: {analysis['coverage']:.2%}")
    print(f"  Más visitados: {analysis['most_visited']}")

    # Ejemplo 2: Tiempos
    print("\n--- EJEMPLO 2: HITTING TIME ---")

    hitting_ab = rw.estimate_hitting_time("A", "B", num_trials=1000)
    hitting_ac = rw.estimate_hitting_time("A", "C", num_trials=1000)

    print(f"Tiempo esperado A -> B: {hitting_ab:.2f} pasos")
    print(f"Tiempo esperado A -> C: {hitting_ac:.2f} pasos")

    # Ejemplo 3: Red de documentos
    print("\n--- EJEMPLO 3: RED DE DOCUMENTOS ---")

    doc_walk = RandomWalk()

    documents = {
        "intro": ["basics", "overview"],
        "basics": ["intro", "intermediate", "examples"],
        "intermediate": ["basics", "advanced"],
        "advanced": ["intermediate", "research"],
        "overview": ["intro", "basics"],
        "examples": ["basics", "intermediate"],
        "research": ["advanced"],
    }

    doc_walk.build_from_dict(documents)

    print("Red de documentos creada:")
    print(f"  {len(doc_walk.nodes)} documentos")
    print(f"  {sum(len(v) for v in documents.values())} links")

    print("\n--- SIMULACIÓN DE USUARIO ---")

    user_path = doc_walk.simulate_user_navigation(
        "intro", num_clicks=30, return_probability=0.15
    )

    print(f"Usuario comienza en 'intro'")
    print(f"Realiza 30 clicks")
    print(f"Probabilidad de retorno: 15%")

    print(f"\nPrimeros 15 clicks:")
    print(" -> ".join(user_path[:15]))

    nav_analysis = doc_walk.analyze_walk(user_path)

    print(f"\nAnálisis de navegación:")
    print(f"  Documentos visitados: {nav_analysis['unique_nodes']}/{len(doc_walk.nodes)}")
    print(f"  Cobertura: {nav_analysis['coverage']:.1%}")

    print(f"\n  Top 5 documentos más visitados:")
    for doc, count in nav_analysis["most_visited"]:
        freq = count / nav_analysis["length"]
        print(f"    {doc:12}: {count:2d} veces ({freq:.1%})")

    # Ejemplo 4: Cover Time
    print("\n--- EJEMPLO 4: COVER TIME ---")

    cover = doc_walk.estimate_cover_time("intro", num_trials=50)

    print(f"Cover Time estimado: {cover:.1f} pasos")
    print(f"(Tiempo para visitar todos los {len(doc_walk.nodes)} documentos)")

    # Ejemplo 5: Mixing Time
    print("\n--- EJEMPLO 5: MIXING TIME ---")

    mixing = doc_walk.estimate_mixing_time(num_trials=50)

    print(f"Mixing Time estimado: {mixing} pasos")
    print(f"(Tiempo para alcanzar distribución estacionaria)")

    print("\n--- ESTADÍSTICAS DEL GRAFO ---")
    stats = doc_walk.get_statistics()

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\n--- EJEMPLO 6: ANÁLISIS DE MÚLTIPLES WALKS ---")

    print("Realizando 100 caminatas desde 'intro'...")

    all_paths = []
    for _ in range(100):
        path = doc_walk.walk("intro", 50)
        all_paths.append(path)

    all_visits = Counter()
    for path in all_paths:
        all_visits.update(path)

    total_visits = sum(all_visits.values())

    print(f"\nDistribución de visitas (100 walks de 50 pasos):")
    for doc in sorted(doc_walk.nodes):
        count = all_visits[doc]
        freq = count / total_visits
        bar = "█" * int(freq * 100)
        print(f"  {doc:12}: {freq:.3f} {bar}")

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)
