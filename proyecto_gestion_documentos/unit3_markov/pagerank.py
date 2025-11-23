"""
PageRank Algorithm
Unidad 3: Cadenas de Markov - PageRank

PageRank es un algoritmo que usa cadenas de Markov para
calcular la "importancia" de nodos en un grafo.

Desarrollado por Larry Page y Sergey Brin (Google).

Aplicación: Determinar importancia de documentos en una red
"""

import numpy as np
from collections import defaultdict


class PageRank:
    """
    Implementación del algoritmo PageRank.

    PageRank modela un "navegador aleatorio" que:
    1. Con probabilidad d, sigue un link aleatorio
    2. Con probabilidad (1-d), salta a cualquier página aleatoria

    Fórmula:
    PR(i) = (1-d)/N + d * Σ(PR(j)/L(j))

    Donde:
    - d = damping factor (típicamente 0.85)
    - N = número total de nodos
    - L(j) = número de links salientes del nodo j

    Atributos:
    ----------
    nodes : list
        Lista de nodos
    adjacency : dict
        Grafo como dict: nodo → [vecinos]
    damping_factor : float
        Factor de amortiguamiento
    """

    def __init__(self, damping_factor=0.85):
        """
        Inicializa PageRank.

        Parámetros:
        -----------
        damping_factor : float
            Factor de amortiguamiento (0 a 1)
            Típicamente 0.85

        Ejemplo:
        --------
        >>> pr = PageRank(damping_factor=0.85)
        """
        self.damping_factor = damping_factor
        self.adjacency = defaultdict(list)
        self.nodes = []
        self.pagerank_scores = {}

    def add_edge(self, from_node, to_node):
        """
        Añade un enlace dirigido entre nodos.

        Parámetros:
        -----------
        from_node : str
            Nodo origen
        to_node : str
            Nodo destino

        Ejemplo:
        --------
        >>> pr.add_edge("doc1", "doc2")
        """
        if from_node not in self.nodes:
            self.nodes.append(from_node)
        if to_node not in self.nodes:
            self.nodes.append(to_node)

        self.adjacency[from_node].append(to_node)

    def build_from_dict(self, graph_dict):
        """
        Construye el grafo desde un diccionario.

        Parámetros:
        -----------
        graph_dict : dict
            Diccionario nodo → [vecinos]

        Ejemplo:
        --------
        >>> graph = {
        ...     'A': ['B', 'C'],
        ...     'B': ['C'],
        ...     'C': ['A']
        ... }
        >>> pr.build_from_dict(graph)
        """
        self.adjacency = defaultdict(list, graph_dict)
        self.nodes = list(graph_dict.keys())

    def compute_pagerank(self, max_iterations=100, tolerance=1e-6):
        """
        Calcula PageRank usando el método iterativo.

        Algoritmo:
        1. Inicializar todos los nodos con 1/N
        2. Iterar:
           PR_new(i) = (1-d)/N + d * Σ(PR(j)/L(j))
        3. Hasta convergencia

        Parámetros:
        -----------
        max_iterations : int
            Número máximo de iteraciones
        tolerance : float
            Tolerancia para convergencia

        Retorna:
        --------
        dict
            Diccionario nodo → PageRank score

        Ejemplo:
        --------
        >>> scores = pr.compute_pagerank()
        >>> print(scores['A'])
        0.387
        """
        N = len(self.nodes)

        if N == 0:
            return {}

        # Inicializar con distribución uniforme
        pagerank = {node: 1.0 / N for node in self.nodes}

        # Calcular número de links salientes por nodo
        outgoing_links = {
            node: len(self.adjacency[node]) if len(self.adjacency[node]) > 0 else N
            for node in self.nodes
        }

        # Iterar hasta convergencia
        for iteration in range(max_iterations):
            new_pagerank = {}

            for node in self.nodes:
                # Término de teleportación
                rank = (1 - self.damping_factor) / N

                # Sumar contribuciones de nodos que apuntan a este
                for source in self.nodes:
                    if node in self.adjacency[source]:
                        rank += (
                            self.damping_factor * pagerank[source] / outgoing_links[source]
                        )
                    elif len(self.adjacency[source]) == 0:
                        # Nodo sin salidas: distribuir uniformemente
                        rank += self.damping_factor * pagerank[source] / N

                new_pagerank[node] = rank

            # Verificar convergencia
            diff = sum(abs(new_pagerank[node] - pagerank[node]) for node in self.nodes)

            pagerank = new_pagerank

            if diff < tolerance:
                break

        # Normalizar (opcional, suma debe ser ~1)
        total = sum(pagerank.values())
        pagerank = {node: score / total for node, score in pagerank.items()}

        self.pagerank_scores = pagerank
        return pagerank

    def get_top_nodes(self, n=10):
        """
        Obtiene los top N nodos por PageRank.

        Parámetros:
        -----------
        n : int
            Número de nodos a retornar

        Retorna:
        --------
        list
            Lista de tuplas (nodo, score)
        """
        if not self.pagerank_scores:
            self.compute_pagerank()

        sorted_nodes = sorted(
            self.pagerank_scores.items(), key=lambda x: x[1], reverse=True
        )

        return sorted_nodes[:n]

    def get_statistics(self):
        """
        Obtiene estadísticas del grafo y PageRank.

        Retorna:
        --------
        dict
            Estadísticas
        """
        if not self.pagerank_scores:
            self.compute_pagerank()

        scores = list(self.pagerank_scores.values())

        # Calcular grado de entrada y salida
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)

        for source, targets in self.adjacency.items():
            out_degree[source] = len(targets)
            for target in targets:
                in_degree[target] += 1

        return {
            "num_nodes": len(self.nodes),
            "num_edges": sum(len(targets) for targets in self.adjacency.values()),
            "avg_pagerank": np.mean(scores),
            "max_pagerank": np.max(scores),
            "min_pagerank": np.min(scores),
            "std_pagerank": np.std(scores),
            "avg_out_degree": np.mean(list(out_degree.values())) if out_degree else 0,
            "avg_in_degree": np.mean(list(in_degree.values())) if in_degree else 0,
        }

    def visualize(self, top_n=10):
        """
        Visualiza los resultados de PageRank.

        Parámetros:
        -----------
        top_n : int
            Número de nodos top a mostrar
        """
        print("\n" + "=" * 60)
        print(f"PAGERANK - TOP {top_n} NODOS")
        print("=" * 60)

        top_nodes = self.get_top_nodes(top_n)

        for i, (node, score) in enumerate(top_nodes, 1):
            bar_length = int(score * 100)
            bar = "█" * bar_length
            print(f"{i:2d}. {node:20} {score:.6f} {bar}")

        print("=" * 60)


class DocumentNetwork:
    """
    Red de documentos para análisis con PageRank.

    Modela documentos y sus referencias/citas.
    """

    def __init__(self):
        """Inicializa la red de documentos."""
        self.pagerank = PageRank(damping_factor=0.85)
        self.document_metadata = {}

    def add_document(self, doc_id, metadata=None):
        """
        Añade un documento a la red.

        Parámetros:
        -----------
        doc_id : str
            ID del documento
        metadata : dict, optional
            Metadatos del documento
        """
        if metadata:
            self.document_metadata[doc_id] = metadata

    def add_reference(self, from_doc, to_doc):
        """
        Añade una referencia de un documento a otro.

        Parámetros:
        -----------
        from_doc : str
            Documento que hace referencia
        to_doc : str
            Documento referenciado
        """
        self.pagerank.add_edge(from_doc, to_doc)

    def compute_importance(self):
        """
        Calcula la importancia de cada documento.

        Retorna:
        --------
        dict
            Diccionario doc_id → score de importancia
        """
        return self.pagerank.compute_pagerank()

    def get_most_important(self, n=10):
        """
        Obtiene los documentos más importantes.

        Parámetros:
        -----------
        n : int
            Número de documentos

        Retorna:
        --------
        list
            Lista de tuplas (doc_id, score)
        """
        return self.pagerank.get_top_nodes(n)

    def get_document_report(self, doc_id):
        """
        Genera un reporte de un documento.

        Parámetros:
        -----------
        doc_id : str
            ID del documento

        Retorna:
        --------
        dict
            Reporte del documento
        """
        scores = self.pagerank.pagerank_scores

        # Referencias salientes
        outgoing = self.pagerank.adjacency[doc_id]

        # Referencias entrantes
        incoming = [
            node for node in self.pagerank.nodes if doc_id in self.pagerank.adjacency[node]
        ]

        return {
            "document_id": doc_id,
            "pagerank_score": scores.get(doc_id, 0),
            "references_out": len(outgoing),
            "references_in": len(incoming),
            "metadata": self.document_metadata.get(doc_id, {}),
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("PAGERANK - DEMOSTRACIÓN")
    print("=" * 60)

    # Ejemplo 1: Grafo simple
    print("\n--- EJEMPLO 1: GRAFO SIMPLE ---")

    pr = PageRank(damping_factor=0.85)

    # Crear grafo de ejemplo
    # A → B, C
    # B → C
    # C → A
    # D → B, C

    pr.add_edge("A", "B")
    pr.add_edge("A", "C")
    pr.add_edge("B", "C")
    pr.add_edge("C", "A")
    pr.add_edge("D", "B")
    pr.add_edge("D", "C")

    print("Grafo:")
    print("  A → B, C")
    print("  B → C")
    print("  C → A")
    print("  D → B, C")

    # Calcular PageRank
    scores = pr.compute_pagerank()

    print("\nPageRank scores:")
    for node in sorted(scores.keys()):
        print(f"  {node}: {scores[node]:.4f}")

    # Visualizar
    pr.visualize(top_n=4)

    # Ejemplo 2: Red de documentos
    print("\n--- EJEMPLO 2: RED DE DOCUMENTOS ---")

    doc_network = DocumentNetwork()

    # Añadir documentos
    documents = {
        "paper_ml": {"title": "Machine Learning Basics", "year": 2020},
        "paper_dl": {"title": "Deep Learning", "year": 2021},
        "paper_cnn": {"title": "CNNs Explained", "year": 2021},
        "paper_nlp": {"title": "NLP with Transformers", "year": 2022},
        "paper_ai": {"title": "AI Overview", "year": 2019},
    }

    for doc_id, metadata in documents.items():
        doc_network.add_document(doc_id, metadata)

    # Añadir referencias (citas)
    # paper_ai → todos (es overview)
    doc_network.add_reference("paper_ai", "paper_ml")
    doc_network.add_reference("paper_ai", "paper_dl")
    doc_network.add_reference("paper_ai", "paper_nlp")

    # paper_dl → paper_ml (DL cita ML)
    doc_network.add_reference("paper_dl", "paper_ml")

    # paper_cnn → paper_dl (CNN cita DL)
    doc_network.add_reference("paper_cnn", "paper_dl")

    # paper_nlp → paper_ml, paper_dl
    doc_network.add_reference("paper_nlp", "paper_ml")
    doc_network.add_reference("paper_nlp", "paper_dl")

    print("Red de documentos creada")
    print("  5 papers")
    print("  7 referencias")

    # Calcular importancia
    importance = doc_network.compute_importance()

    print("\n--- DOCUMENTOS MÁS IMPORTANTES ---")
    top_docs = doc_network.get_most_important(n=5)

    for i, (doc_id, score) in enumerate(top_docs, 1):
        metadata = documents[doc_id]
        print(f"\n{i}. {doc_id}")
        print(f"   Título: {metadata['title']}")
        print(f"   Año: {metadata['year']}")
        print(f"   PageRank: {score:.4f}")

        report = doc_network.get_document_report(doc_id)
        print(f"   Citas recibidas: {report['references_in']}")
        print(f"   Referencias: {report['references_out']}")

    # Estadísticas
    print("\n--- ESTADÍSTICAS DE LA RED ---")
    stats = doc_network.pagerank.get_statistics()

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)
