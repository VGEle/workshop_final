"""
Probar motor de búsqueda con datos reales
"""

import json
import time
from pathlib import Path
from search_engine import SearchEngine

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "unit1_documents.json"


def cargar_documentos(archivo):
    """Carga documentos desde archivo JSON"""
    archivo = Path(archivo)
    print(f"Cargando documentos desde {archivo}...")
    with archivo.open("r", encoding="utf-8") as f:
        documentos = json.load(f)
    print(f"✓ Cargados {len(documentos)} documentos")
    return documentos


def probar_motor_busqueda():
    """Prueba el motor de búsqueda con datos reales"""

    print("\n" + "=" * 60)
    print("PRUEBA DE MOTOR DE BÚSQUEDA CON DATOS REALES")
    print("=" * 60 + "\n")

    # Cargar documentos
    documentos = cargar_documentos(DATA_FILE)

    # Crear motor de búsqueda
    print("\nCreando motor de búsqueda...")
    engine = SearchEngine()

    # Añadir documentos
    print("Añadiendo documentos al motor...")
    inicio = time.time()

    for i, doc in enumerate(documentos):
        engine.add_document(doc)

        if (i + 1) % 500 == 0:
            print(f"  Procesados {i + 1}/{len(documentos)} documentos...")

    tiempo_indexacion = time.time() - inicio
    print(f"✓ Indexados {len(documentos)} documentos en {tiempo_indexacion:.2f} segundos")

    # Estadísticas
    print("\n--- ESTADÍSTICAS DEL MOTOR ---")
    stats = engine.get_statistics()
    print(f"  Documentos totales: {stats['num_documents']}")
    print(f"  Vocabulario único: {stats['num_unique_words']} palabras")

    # Realizar búsquedas de prueba
    print("\n--- BÚSQUEDAS DE PRUEBA ---")

    consultas = [
        "report annual",
        "project management",
        "financial analysis",
        "technical documentation",
        "meeting notes",
    ]

    for consulta in consultas:
        print(f"\nBúsqueda: '{consulta}'") 

        inicio = time.time()
        resultados = engine.search(consulta, max_results=5)
        tiempo_busqueda = time.time() - inicio

        print(f"   Tiempo: {tiempo_busqueda:.4f} segundos")
        print(f"   Resultados encontrados: {len(resultados)}")

        if resultados:
            print("   Top 3 resultados:")
            for i, (doc, score) in enumerate(resultados[:3], 1):
                titulo = (
                    doc["title"][:50] + "..." if len(doc["title"]) > 50 else doc["title"]
                )
                print(f"     {i}. [Score: {score:.2f}] {titulo}")

    print("\n" + "=" * 60)
    print("✓ PRUEBA COMPLETADA")
    print("=" * 60)


if __name__ == "__main__":
    probar_motor_busqueda()
