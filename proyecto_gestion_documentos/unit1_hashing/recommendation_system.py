"""
Sistema de Recomendaciones Aleatorizado
Unidad 1: Hashing y Algoritmos Aleatorizados

Sugiere documentos relevantes basándose en:
- Historial de acceso del usuario
- Similitud de tags
- Algoritmos aleatorizados (randomized algorithms)
"""

import random
from collections import Counter, defaultdict


class RecommendationSystem:
    """
    Sistema de recomendaciones que usa algoritmos aleatorizados.

    Estrategias implementadas:
    1. Recomendación por tags similares
    2. Recomendación por popularidad
    3. Exploración aleatoria (exploration)
    4. Filtrado colaborativo simple

    Atributos:
    ----------
    hash_table : HashTable
        Referencia a la tabla hash de documentos
    user_history : dict
        Historial de acceso por usuario
    document_access_count : dict
        Contador de accesos por documento
    """

    def __init__(self, hash_table):
        """
        Inicializa el sistema de recomendaciones.

        Parámetros:
        -----------
        hash_table : HashTable
            Tabla hash con los documentos
        """
        self.hash_table = hash_table
        self.user_history = defaultdict(list)  # user_id → [doc_ids]
        self.document_access_count = Counter()  # doc_id → count
        self.tag_index = defaultdict(list)  # tag → [doc_ids]

        # Construir índice de tags
        self._build_tag_index()

    def _build_tag_index(self):
        """
        Construye un índice de documentos por tag.

        Recorre todos los documentos en la tabla hash y
        crea un mapeo de tag → documentos que tienen ese tag.

        Complejidad: O(n*m) donde n es número de documentos
        y m es promedio de tags por documento
        """
        print("Construyendo índice de tags...")

        # Iterar sobre todas las posiciones de la tabla hash
        for bucket in self.hash_table.table:
            for document in bucket:
                doc_id = document["_id"]

                # Añadir documento a cada uno de sus tags
                if "tags" in document:
                    for tag in document["tags"]:
                        tag_lower = tag.lower()
                        self.tag_index[tag_lower].append(doc_id)

        print(f"✓ Índice construido: {len(self.tag_index)} tags únicos")

    def record_access(self, user_id, document_id):
        """
        Registra que un usuario accedió a un documento.

        Actualiza:
        1. Historial del usuario
        2. Contador de popularidad del documento

        Parámetros:
        -----------
        user_id : str
            ID del usuario
        document_id : str
            ID del documento accedido

        Ejemplo:
        --------
        >>> rec_sys.record_access("user123", "doc456")
        """
        self.user_history[user_id].append(document_id)
        self.document_access_count[document_id] += 1

    def recommend_by_tags(self, document_id, num_recommendations=5):
        """
        Recomienda documentos con tags similares.

        Algoritmo:
        1. Obtener tags del documento actual
        2. Encontrar otros documentos con esos tags
        3. Calcular similitud (número de tags en común)
        4. Retornar top N más similares

        Parámetros:
        -----------
        document_id : str
            ID del documento base
        num_recommendations : int
            Número de recomendaciones

        Retorna:
        --------
        list
            Lista de tuplas (documento, similitud)

        Complejidad: O(n*m) donde n es número de docs con tags similares
        y m es número promedio de tags
        """
        # Obtener documento base
        base_doc = self.hash_table.search(document_id)

        if base_doc is None or "tags" not in base_doc:
            return []

        base_tags = set(tag.lower() for tag in base_doc["tags"])

        # Encontrar documentos candidatos
        candidate_docs = set()
        for tag in base_tags:
            if tag in self.tag_index:
                candidate_docs.update(self.tag_index[tag])

        # Remover el documento base
        candidate_docs.discard(document_id)

        # Calcular similitud para cada candidato
        similarities = []

        for doc_id in candidate_docs:
            doc = self.hash_table.search(doc_id)

            if doc is None or "tags" not in doc:
                continue

            # Calcular similitud de Jaccard
            doc_tags = set(tag.lower() for tag in doc["tags"])

            # Intersección / Unión
            intersection = len(base_tags & doc_tags)
            union = len(base_tags | doc_tags)

            if union > 0:
                similarity = intersection / union
                similarities.append((doc, similarity))

        # Ordenar por similitud y retornar top N
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_recommendations]

    def recommend_popular(self, num_recommendations=5, exclude_ids=None):
        """
        Recomienda documentos más populares (más accedidos).

        Parámetros:
        -----------
        num_recommendations : int
            Número de recomendaciones
        exclude_ids : set
            IDs de documentos a excluir

        Retorna:
        --------
        list
            Lista de tuplas (documento, num_accesos)
        """
        if exclude_ids is None:
            exclude_ids = set()

        # Obtener documentos más populares
        popular = []

        for doc_id, count in self.document_access_count.most_common(100):
            if doc_id not in exclude_ids:
                doc = self.hash_table.search(doc_id)
                if doc is not None:
                    popular.append((doc, count))

                    if len(popular) >= num_recommendations:
                        break

        return popular

    def recommend_random_exploration(self, num_recommendations=5, exclude_ids=None):
        """
        Recomendaciones aleatorias para exploración.

        Usa algoritmo de muestreo aleatorio para sugerir
        documentos que el usuario podría no haber considerado.

        Esto implementa el concepto de "exploration" en
        sistemas de recomendación (exploration vs exploitation).

        Parámetros:
        -----------
        num_recommendations : int
            Número de recomendaciones
        exclude_ids : set
            IDs a excluir

        Retorna:
        --------
        list
            Lista de documentos aleatorios
        """
        if exclude_ids is None:
            exclude_ids = set()

        # Recolectar todos los documentos disponibles
        all_docs = []

        for bucket in self.hash_table.table:
            for doc in bucket:
                if doc["_id"] not in exclude_ids:
                    all_docs.append(doc)

        # Muestreo aleatorio sin reemplazo
        if len(all_docs) <= num_recommendations:
            return all_docs

        return random.sample(all_docs, num_recommendations)

    def recommend_hybrid(
        self, user_id=None, document_id=None, num_recommendations=5, exploitation_ratio=0.7
    ):
        """
        Sistema híbrido de recomendaciones.

        Combina múltiples estrategias:
        - exploitation_ratio: proporción de recomendaciones basadas
          en similitud/popularidad
        - (1 - exploitation_ratio): proporción de exploración aleatoria

        Esta es una implementación de algoritmos aleatorizados donde
        balanceamos exploitation (usar lo que sabemos) con
        exploration (descubrir cosas nuevas).

        Parámetros:
        -----------
        user_id : str
            ID del usuario (opcional)
        document_id : str
            ID del documento actual (opcional)
        num_recommendations : int
            Número total de recomendaciones
        exploitation_ratio : float
            Proporción de recomendaciones no aleatorias (0.0 a 1.0)

        Retorna:
        --------
        list
            Lista de tuplas (documento, score, tipo)
            donde tipo es 'similar', 'popular' o 'exploration'

        Ejemplo:
        --------
        >>> # 70% similares/populares, 30% aleatorias
        >>> recs = rec_sys.recommend_hybrid(
        ...     user_id="user1",
        ...     document_id="doc123",
        ...     num_recommendations=10,
        ...     exploitation_ratio=0.7
        ... )
        """
        recommendations = []
        exclude_ids = set()

        # Calcular cuántas de cada tipo
        num_exploit = int(num_recommendations * exploitation_ratio)
        num_explore = num_recommendations - num_exploit

        # 1. Recomendaciones por similitud (si hay documento base)
        if document_id:
            similar = self.recommend_by_tags(document_id, num_exploit // 2)

            for doc, similarity in similar:
                recommendations.append((doc, similarity, "similar"))
                exclude_ids.add(doc["_id"])

        # 2. Recomendaciones por popularidad
        remaining_exploit = num_exploit - len(recommendations)
        if remaining_exploit > 0:
            popular = self.recommend_popular(remaining_exploit, exclude_ids)

            for doc, count in popular:
                # Normalizar count a [0, 1]
                score = min(count / 100.0, 1.0)
                recommendations.append((doc, score, "popular"))
                exclude_ids.add(doc["_id"])

        # 3. Exploración aleatoria
        if num_explore > 0:
            random_docs = self.recommend_random_exploration(num_explore, exclude_ids)

            for doc in random_docs:
                recommendations.append((doc, 0.5, "exploration"))

        # Mezclar aleatoriamente
        random.shuffle(recommendations)

        return recommendations[:num_recommendations]

    def get_user_statistics(self, user_id):
        """
        Obtiene estadísticas de un usuario.

        Retorna:
        --------
        dict
            Estadísticas del usuario
        """
        history = self.user_history.get(user_id, [])

        if not history:
            return {"total_accesses": 0, "unique_documents": 0, "favorite_tags": []}

        # Contar tags más frecuentes
        tag_counter = Counter()

        for doc_id in set(history):  # Únicos
            doc = self.hash_table.search(doc_id)
            if doc and "tags" in doc:
                tag_counter.update(doc["tags"])

        return {
            "total_accesses": len(history),
            "unique_documents": len(set(history)),
            "favorite_tags": tag_counter.most_common(5),
        }


# Ejemplo de uso
if __name__ == "__main__":
    from hash_table import HashTable

    print("=" * 60)
    print("SISTEMA DE RECOMENDACIONES - EJEMPLO")
    print("=" * 60)

    # Crear tabla hash y añadir documentos
    ht = HashTable(size=100)

    documents = [
        {
            "_id": "doc1",
            "title": "Python Tutorial",
            "tags": ["python", "programming", "tutorial"],
        },
        {
            "_id": "doc2",
            "title": "Python Web Development",
            "tags": ["python", "web", "flask"],
        },
        {
            "_id": "doc3",
            "title": "Java Basics",
            "tags": ["java", "programming", "tutorial"],
        },
        {
            "_id": "doc4",
            "title": "Data Science",
            "tags": ["python", "data", "science"],
        },
        {
            "_id": "doc5",
            "title": "Machine Learning",
            "tags": ["python", "ml", "ai"],
        },
    ]

    for doc in documents:
        ht.insert(doc)

    # Crear sistema de recomendaciones
    rec_sys = RecommendationSystem(ht)

    # Simular accesos de usuarios
    print("\nSimulando accesos de usuarios...")
    rec_sys.record_access("user1", "doc1")
    rec_sys.record_access("user1", "doc2")
    rec_sys.record_access("user2", "doc1")
    rec_sys.record_access("user2", "doc3")

    # Recomendaciones por similitud
    print("\n--- RECOMENDACIONES POR SIMILITUD ---")
    print("Basadas en: 'Python Tutorial' (doc1)")
    similar = rec_sys.recommend_by_tags("doc1", num_recommendations=3)

    for doc, similarity in similar:
        print(f"  • [{similarity:.2f}] {doc['title']}")
        print(f"    Tags: {', '.join(doc['tags'])}")

    # Recomendaciones híbridas
    print("\n--- RECOMENDACIONES HÍBRIDAS ---")
    print("70% similares/populares, 30% exploración")
    hybrid = rec_sys.recommend_hybrid(
        user_id="user1",
        document_id="doc1",
        num_recommendations=5,
        exploitation_ratio=0.7,
    )

    for doc, score, tipo in hybrid:
        print(f"  • [{tipo}] {doc['title']} (score: {score:.2f})")

    # Estadísticas de usuario
    print("\n--- ESTADÍSTICAS DE USUARIO ---")
    stats = rec_sys.get_user_statistics("user1")
    print("User: user1")
    print(f"  Total de accesos: {stats['total_accesses']}")
    print(f"  Documentos únicos: {stats['unique_documents']}")
    print(f"  Tags favoritos: {[tag for tag, count in stats['favorite_tags']]}")

    print("\n" + "=" * 60)
    print("✓ EJEMPLO COMPLETADO")
    print("=" * 60)
