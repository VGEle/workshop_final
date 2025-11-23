"""
Sistema Completo de Gestión de Documentos - Unidad 1
Integra: Hash Table, Search Engine, Recommendation System
"""

import json
import time
from hash_table import HashTable
from search_engine import SearchEngine
from recommendation_system import RecommendationSystem


class DocumentManagementSystem:
    """
    Sistema completo de gestión de documentos.

    Integra todos los componentes de la Unidad 1:
    - Almacenamiento con tabla hash
    - Motor de búsqueda
    - Sistema de recomendaciones
    - Seguimiento de interacciones
    """

    def __init__(self):
        """Inicializa el sistema completo"""
        print("Inicializando Sistema de Gestión de Documentos...")

        self.search_engine = SearchEngine()
        self.recommendation_system = None  # Se inicializa después de cargar datos
        self.loaded = False

        print("Sistema inicializado")

    def load_documents(self, filepath):
        """
        Carga documentos desde un archivo JSON.

        Parámetros:
        -----------
        filepath : str
            Ruta al archivo JSON con documentos

        Retorna:
        --------
        int
            Número de documentos cargados
        """
        print(f"\nCargando documentos desde {filepath}...")
        inicio = time.time()

        with open(filepath, "r", encoding="utf-8") as f:
            documents = json.load(f)

        print(f"Archivo cargado: {len(documents)} documentos")

        # Añadir documentos al motor de búsqueda
        print("Indexando documentos...")

        for i, doc in enumerate(documents):
            self.search_engine.add_document(doc)

            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(documents)}...")

        # Inicializar sistema de recomendaciones
        print("Inicializando sistema de recomendaciones...")
        self.recommendation_system = RecommendationSystem(self.search_engine.hash_table)

        tiempo_total = time.time() - inicio
        self.loaded = True

        print(f"Sistema cargado en {tiempo_total:.2f} segundos")
        print(f"  Documentos: {len(documents)}")
        print(f"  Vocabulario: {len(self.search_engine.inverted_index)} palabras")

        return len(documents)

    def search(self, query, max_results=10):
        """
        Busca documentos por palabras clave.

        Parámetros:
        -----------
        query : str
            Consulta de búsqueda
        max_results : int
            Número máximo de resultados

        Retorna:
        --------
        list
            Lista de (documento, score)
        """
        if not self.loaded:
            print("ADVERTENCIA: Sistema no cargado. Use load_documents() primero.")
            return []

        return self.search_engine.search(query, max_results)

    def get_recommendations(
        self, user_id=None, document_id=None, num_recommendations=5, strategy="hybrid"
    ):
        """
        Obtiene recomendaciones de documentos.

        Parámetros:
        -----------
        user_id : str
            ID del usuario (opcional)
        document_id : str
            ID del documento actual (opcional)
        num_recommendations : int
            Número de recomendaciones
        strategy : str
            Estrategia: 'hybrid', 'similar', 'popular', 'random'

        Retorna:
        --------
        list
            Lista de recomendaciones
        """
        if not self.loaded:
            print("ADVERTENCIA: Sistema no cargado. Use load_documents() primero.")
            return []

        if strategy == "hybrid":
            return self.recommendation_system.recommend_hybrid(
                user_id, document_id, num_recommendations
            )
        elif strategy == "similar":
            if document_id:
                return self.recommendation_system.recommend_by_tags(
                    document_id, num_recommendations
                )
            else:
                print("ADVERTENCIA: 'similar' requiere document_id")
                return []
        elif strategy == "popular":
            return self.recommendation_system.recommend_popular(num_recommendations)
        elif strategy == "random":
            return self.recommendation_system.recommend_random_exploration(
                num_recommendations
            )
        else:
            print(f"ADVERTENCIA: Estrategia desconocida: {strategy}")
            return []

    def record_user_access(self, user_id, document_id):
        """
        Registra que un usuario accedió a un documento.

        Parámetros:
        -----------
        user_id : str
            ID del usuario
        document_id : str
            ID del documento
        """
        if not self.loaded:
            return

        self.recommendation_system.record_access(user_id, document_id)

    def get_document(self, document_id):
        """
        Obtiene un documento por su ID.

        Parámetros:
        -----------
        document_id : str
            ID del documento

        Retorna:
        --------
        dict or None
            Documento si existe
        """
        if not self.loaded:
            return None

        return self.search_engine.hash_table.search(document_id)

    def get_user_stats(self, user_id):
        """
        Obtiene estadísticas de un usuario.

        Parámetros:
        -----------
        user_id : str
            ID del usuario

        Retorna:
        --------
        dict
            Estadísticas del usuario
        """
        if not self.loaded:
            return {}

        return self.recommendation_system.get_user_statistics(user_id)

    def get_system_stats(self):
        """
        Obtiene estadísticas del sistema.

        Retorna:
        --------
        dict
            Estadísticas generales
        """
        if not self.loaded:
            return {}

        search_stats = self.search_engine.get_statistics()

        return {
            "documents": search_stats["num_documents"],
            "vocabulary": search_stats["num_unique_words"],
            "unique_tags": len(self.recommendation_system.tag_index),
            "total_accesses": sum(
                self.recommendation_system.document_access_count.values()
            ),
            "hash_table": search_stats["hash_table_stats"],
        }

    def interactive_demo(self):
        """
        Demostración interactiva del sistema.
        """
        if not self.loaded:
            print("ADVERTENCIA: Cargue documentos primero con load_documents()")
            return

        print("\n" + "=" * 60)
        print("DEMOSTRACIÓN INTERACTIVA DEL SISTEMA")
        print("=" * 60)

        # Simular usuario
        user_id = "demo_user"

        while True:
            print("\n--- MENÚ ---")
            print("1. Buscar documentos")
            print("2. Ver recomendaciones")
            print("3. Ver estadísticas del sistema")
            print("4. Ver mis estadísticas")
            print("5. Salir")

            opcion = input("\nSeleccione una opción (1-5): ").strip()

            if opcion == "1":
                query = input("Ingrese su búsqueda: ").strip()
                if query:
                    print(f"\nBuscando: '{query}'")
                    resultados = self.search(query, max_results=5)

                    if resultados:
                        print(f"\nEncontrados {len(resultados)} resultados:")
                        for i, (doc, score) in enumerate(resultados, 1):
                            print(f"\n{i}. [Score: {score:.2f}]")
                            print(f"   ID: {doc['_id'][:20]}...")
                            print(f"   Título: {doc['title'][:60]}...")

                            # Registrar acceso al primer resultado
                            if i == 1:
                                self.record_user_access(user_id, doc["_id"])
                                print("   Acceso registrado")
                    else:
                        print("No se encontraron resultados")

            elif opcion == "2":
                print("\nEstrategias disponibles:")
                print("  h - Híbrida (recomendada)")
                print("  p - Por popularidad")
                print("  r - Aleatoria")

                estrategia = input("Seleccione estrategia (h/p/r): ").strip().lower()

                strategy_map = {"h": "hybrid", "p": "popular", "r": "random"}
                strategy = strategy_map.get(estrategia, "hybrid")

                print(f"\nRecomendaciones ({strategy}):")
                recs = self.get_recommendations(
                    user_id=user_id, num_recommendations=5, strategy=strategy
                )

                if recs:
                    for i, item in enumerate(recs, 1):
                        if len(item) == 3:  # hybrid
                            doc, score, tipo = item
                            print(f"\n{i}. [{tipo}] (score: {score:.2f})")
                        elif len(item) == 2:  # similar/popular
                            doc, score = item
                            print(f"\n{i}. (score: {score:.2f})")
                        else:
                            doc = item
                            print(f"\n{i}.")

                        print(f"   Título: {doc['title'][:60]}...")
                else:
                    print("No hay recomendaciones disponibles")

            elif opcion == "3":
                print("\n--- ESTADÍSTICAS DEL SISTEMA ---")
                stats = self.get_system_stats()
                for key, value in stats.items():
                    if key != "hash_table":
                        print(f"  {key}: {value}")

            elif opcion == "4":
                print(f"\n--- ESTADÍSTICAS DE {user_id} ---")
                stats = self.get_user_stats(user_id)
                for key, value in stats.items():
                    print(f"  {key}: {value}")

            elif opcion == "5":
                print("\nSaliendo del sistema interactivo.")
                break

            else:
                print("ADVERTENCIA: Opción inválida")


# Script principal
if __name__ == "__main__":
    print("=" * 60)
    print("SISTEMA INTEGRADO DE GESTIÓN DE DOCUMENTOS")
    print("Unidad 1: Hashing y Algoritmos Aleatorizados")
    print("=" * 60)

    # Crear sistema
    dms = DocumentManagementSystem()

    # Cargar documentos
    dms.load_documents("../data/unit1_documents.json")

    # Mostrar estadísticas iniciales
    print("\n--- ESTADÍSTICAS DEL SISTEMA ---")
    stats = dms.get_system_stats()
    print(f"  Total de documentos: {stats['documents']}")
    print(f"  Vocabulario único: {stats['vocabulary']} palabras")
    print(f"  Tags únicos: {stats['unique_tags']}")

    # Demostración rápida
    print("\n--- DEMOSTRACIÓN RÁPIDA ---")

    # Búsqueda
    print("\n1. Búsqueda de documentos:")
    resultados = dms.search("report financial", max_results=3)
    for i, (doc, score) in enumerate(resultados, 1):
        print(f"   {i}. [Score: {score:.2f}] {doc['title'][:50]}...")

    # Registrar accesos
    if resultados:
        doc_id = resultados[0][0]["_id"]
        dms.record_user_access("user123", doc_id)
        print(f"\n2. Acceso registrado para user123 -> {doc_id[:20]}...")

    # Recomendaciones
    print("\n3. Recomendaciones híbridas:")
    recs = dms.get_recommendations(
        user_id="user123",
        document_id=doc_id if resultados else None,
        num_recommendations=3,
        strategy="hybrid",
    )

    for i, (doc, score, tipo) in enumerate(recs, 1):
        print(f"   {i}. [{tipo}] {doc['title'][:50]}...")

    print("\n" + "=" * 60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("=" * 60)

    # Opción interactiva
    print("\n¿Desea probar el modo interactivo? (s/n): ", end="")
    respuesta = input().strip().lower()

    if respuesta == "s":
        dms.interactive_demo()
