"""
Motor de Búsqueda por Palabras Clave
Unidad 1: Hashing y Algoritmos Aleatorizados

Permite buscar documentos usando palabras clave y rankear
los resultados por relevancia.
"""

from hash_table import HashTable
import re
from collections import Counter


class SearchEngine:
    """
    Motor de búsqueda que permite encontrar documentos por palabras clave.

    Características:
    - Búsqueda por título, contenido y tags
    - Ranking por relevancia
    - Búsqueda case-insensitive
    - Soporte para múltiples palabras clave

    Atributos:
    ----------
    hash_table : HashTable
        Tabla hash que almacena los documentos
    inverted_index : dict
        Índice invertido: palabra → lista de document IDs
    """

    def __init__(self):
        """
        Inicializa el motor de búsqueda.

        Complejidad Temporal: O(1)
        Complejidad Espacial: O(1)
        """
        self.hash_table = HashTable(size=2500)
        self.inverted_index = {}  # palabra → [doc_ids]
        self.num_documents = 0

    def _normalize_text(self, text):
        """
        Normaliza un texto para búsqueda.

        Proceso:
        1. Convierte a minúsculas
        2. Remueve puntuación
        3. Divide en palabras
        4. Remueve palabras vacías comunes

        Parámetros:
        -----------
        text : str
            Texto a normalizar

        Retorna:
        --------
        list
            Lista de palabras normalizadas

        Ejemplo:
        --------
        >>> engine._normalize_text("Hello, World!")
        ['hello', 'world']
        """
        # Convertir a minúsculas
        text = text.lower()

        # Remover puntuación y dividir en palabras
        words = re.findall(r"\b\w+\b", text)

        # Palabras vacías comunes (stop words) a ignorar
        stop_words = {
            "el",
            "la",
            "de",
            "que",
            "y",
            "a",
            "en",
            "un",
            "ser",
            "se",
            "no",
            "por",
            "con",
            "su",
            "para",
            "como",
            "es",
            "al",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "was",
        }

        # Filtrar palabras vacías y palabras muy cortas
        words = [w for w in words if w not in stop_words and len(w) > 2]

        return words

    def add_document(self, document):
        """
        Añade un documento al motor de búsqueda.

        Proceso:
        1. Inserta en la tabla hash
        2. Extrae palabras clave del documento
        3. Actualiza el índice invertido

        Parámetros:
        -----------
        document : dict
            Documento con campos: _id, title, content, tags

        Retorna:
        --------
        bool
            True si se añadió exitosamente

        Complejidad Temporal: O(n) donde n es el número de palabras

        Ejemplo:
        --------
        >>> engine = SearchEngine()
        >>> doc = {
        ...     '_id': 'doc1',
        ...     'title': 'Python Programming',
        ...     'content': 'Learn Python easily',
        ...     'tags': ['python', 'tutorial']
        ... }
        >>> engine.add_document(doc)
        True
        """
        # Insertar en tabla hash
        if not self.hash_table.insert(document):
            return False  # Documento ya existe

        doc_id = document["_id"]
        self.num_documents += 1

        # Extraer todas las palabras del documento
        all_text = []

        # Palabras del título
        if "title" in document:
            all_text.extend(self._normalize_text(document["title"]))

        # Palabras del contenido
        if "content" in document:
            all_text.extend(self._normalize_text(document["content"]))

        # Tags
        if "tags" in document:
            for tag in document["tags"]:
                all_text.extend(self._normalize_text(tag))

        # Actualizar índice invertido
        # Cada palabra apunta a los IDs de documentos que la contienen
        unique_words = set(all_text)  # Evitar duplicados

        for word in unique_words:
            if word not in self.inverted_index:
                self.inverted_index[word] = []
            self.inverted_index[word].append(doc_id)

        return True

    def search(self, query, max_results=10):
        """
        Busca documentos que coincidan con la consulta.

        Algoritmo:
        1. Normalizar la consulta
        2. Encontrar documentos que contengan las palabras
        3. Calcular score de relevancia para cada documento
        4. Ordenar por relevancia
        5. Retornar top N resultados

        Parámetros:
        -----------
        query : str
            Consulta de búsqueda
        max_results : int
            Número máximo de resultados (default: 10)

        Retorna:
        --------
        list
            Lista de tuplas (documento, score) ordenadas por relevancia

        Complejidad Temporal: O(n*m) donde n es el número de palabras
        y m es el número de documentos que las contienen

        Ejemplo:
        --------
        >>> results = engine.search("python programming")
        >>> for doc, score in results:
        ...     print(f"{doc['title']}: {score}")
        Python Programming: 2.5
        Learn Python: 1.8
        """
        # Normalizar consulta
        query_words = self._normalize_text(query)

        if not query_words:
            return []

        # Encontrar documentos candidatos
        candidate_docs = {}  # doc_id → frecuencia de palabras

        for word in query_words:
            if word in self.inverted_index:
                for doc_id in self.inverted_index[word]:
                    if doc_id not in candidate_docs:
                        candidate_docs[doc_id] = 0
                    candidate_docs[doc_id] += 1

        # Si no hay candidatos, retornar lista vacía
        if not candidate_docs:
            return []

        # Calcular score de relevancia para cada documento
        results = []

        for doc_id, word_count in candidate_docs.items():
            # Recuperar documento
            document = self.hash_table.search(doc_id)

            if document is None:
                continue

            # Calcular score de relevancia
            score = self._calculate_relevance_score(document, query_words, word_count)

            results.append((document, score))

        # Ordenar por score (mayor a menor)
        results.sort(key=lambda x: x[1], reverse=True)

        # Retornar top N resultados
        return results[:max_results]

    def _calculate_relevance_score(self, document, query_words, word_count):
        """
        Calcula el score de relevancia de un documento.

        Factores considerados:
        1. Número de palabras de la consulta que aparecen
        2. Frecuencia de las palabras en el documento
        3. Palabras en el título tienen mayor peso
        4. Palabras en tags tienen mayor peso

        Parámetros:
        -----------
        document : dict
            Documento a evaluar
        query_words : list
            Palabras de la consulta
        word_count : int
            Número de palabras de la consulta en el documento

        Retorna:
        --------
        float
            Score de relevancia (mayor = más relevante)

        Ejemplo:
        --------
        Si un documento tiene 2 palabras de la consulta en el título
        y 3 en el contenido, su score será:
        (2 * 3.0) + (3 * 1.0) = 9.0
        """
        score = 0.0

        # Peso base: número de palabras que coinciden
        score += word_count

        # Bonus: palabras en el título (peso x3)
        if "title" in document:
            title_words = self._normalize_text(document["title"])
            title_matches = sum(1 for w in query_words if w in title_words)
            score += title_matches * 3.0

        # Bonus: palabras en tags (peso x2)
        if "tags" in document:
            tag_words = []
            for tag in document["tags"]:
                tag_words.extend(self._normalize_text(tag))
            tag_matches = sum(1 for w in query_words if w in tag_words)
            score += tag_matches * 2.0

        # Bonus: múltiples palabras de la consulta (coherencia)
        if len(query_words) > 1:
            score += word_count * 0.5

        return score

    def search_by_tag(self, tag):
        """
        Busca documentos por un tag específico.

        Parámetros:
        -----------
        tag : str
            Tag a buscar

        Retorna:
        --------
        list
            Lista de documentos que tienen ese tag

        Ejemplo:
        --------
        >>> results = engine.search_by_tag("python")
        >>> print(f"Encontrados {len(results)} documentos")
        """
        tag_normalized = tag.lower()

        if tag_normalized in self.inverted_index:
            doc_ids = self.inverted_index[tag_normalized]
            documents = []

            for doc_id in doc_ids:
                doc = self.hash_table.search(doc_id)
                if doc is not None:
                    documents.append(doc)

            return documents

        return []

    def get_statistics(self):
        """
        Obtiene estadísticas del motor de búsqueda.

        Retorna:
        --------
        dict
            Diccionario con estadísticas
        """
        return {
            "num_documents": self.num_documents,
            "num_unique_words": len(self.inverted_index),
            "hash_table_stats": self.hash_table.get_statistics(),
        }

    def __str__(self):
        """Representación en string"""
        return f"SearchEngine(documents={self.num_documents}, vocabulary={len(self.inverted_index)})"


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("MOTOR DE BÚSQUEDA - EJEMPLO DE USO")
    print("=" * 60)

    # Crear motor de búsqueda
    engine = SearchEngine()

    # Añadir documentos de ejemplo
    documents = [
        {
            "_id": "doc1",
            "title": "Python Programming Tutorial",
            "content": "Learn Python programming from scratch. Python is easy to learn.",
            "tags": ["python", "programming", "tutorial"],
        },
        {
            "_id": "doc2",
            "title": "Java Programming Guide",
            "content": "Complete guide to Java programming language.",
            "tags": ["java", "programming", "guide"],
        },
        {
            "_id": "doc3",
            "title": "Web Development with Python",
            "content": "Build web applications using Python and Flask framework.",
            "tags": ["python", "web", "flask"],
        },
        {
            "_id": "doc4",
            "title": "Data Science Tutorial",
            "content": "Introduction to data science using Python and pandas.",
            "tags": ["python", "data-science", "pandas"],
        },
        {
            "_id": "doc5",
            "title": "Machine Learning Basics",
            "content": "Learn machine learning fundamentals and algorithms.",
            "tags": ["machine-learning", "algorithms"],
        },
    ]

    # Añadir documentos al motor
    print("\nAñadiendo documentos...")
    for doc in documents:
        engine.add_document(doc)
    print(f"✓ Añadidos {len(documents)} documentos")

    # Estadísticas
    print("\n--- ESTADÍSTICAS ---")
    stats = engine.get_statistics()
    print(f"Documentos totales: {stats['num_documents']}")
    print(f"Vocabulario único: {stats['num_unique_words']} palabras")

    # Realizar búsquedas
    print("\n--- BÚSQUEDAS ---")

    queries = [
        "python programming",
        "machine learning",
        "web development",
        "data science",
    ]

    for query in queries:
        print(f"\nBúsqueda: '{query}'")
        results = engine.search(query, max_results=3)

        if results:
            print(f"Encontrados {len(results)} resultados:")
            for i, (doc, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.2f}] {doc['title']}")
        else:
            print("  No se encontraron resultados")

    # Búsqueda por tag
    print("\n--- BÚSQUEDA POR TAG ---")
    tag = "python"
    results = engine.search_by_tag(tag)
    print(f"\nDocumentos con tag '{tag}':")
    for doc in results:
        print(f"  • {doc['title']}")

    print("\n" + "=" * 60)
    print("✓ EJEMPLO COMPLETADO")
    print("=" * 60)
