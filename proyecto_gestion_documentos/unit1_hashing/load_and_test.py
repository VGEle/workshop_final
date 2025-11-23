"""
Cargar documentos JSON y probar la tabla hash
"""

import json
import time
from hash_table import HashTable


def cargar_documentos(archivo):
    """Carga documentos desde un archivo JSON"""
    print(f"Cargando documentos desde {archivo}...")

    with open(archivo, "r", encoding="utf-8") as f:
        documentos = json.load(f)

    print(f"✓ Cargados {len(documentos)} documentos")
    return documentos


def probar_tabla_hash():
    """Prueba la tabla hash con documentos reales"""

    print("\n" + "=" * 60)
    print("PRUEBA DE TABLA HASH CON DATOS REALES")
    print("=" * 60 + "\n")

    # Cargar documentos
    documentos = cargar_documentos("../data/unit1_documents.json")

    # Crear tabla hash
    print("\nCreando tabla hash...")
    ht = HashTable(size=2500)  # Tamaño mayor que número de documentos
    print(f"✓ Tabla hash creada con tamaño {ht.size}")

    # Insertar todos los documentos
    print("\nInsertando documentos en la tabla hash...")
    inicio = time.time()

    for i, doc in enumerate(documentos):
        ht.insert(doc)

        # Mostrar progreso cada 500 documentos
        if (i + 1) % 500 == 0:
            print(f"  Insertados {i + 1}/{len(documentos)} documentos...")

    tiempo_insercion = time.time() - inicio
    print(f"✓ Insertados {len(documentos)} documentos en {tiempo_insercion:.2f} segundos")
    print(f"  Velocidad: {len(documentos)/tiempo_insercion:.0f} docs/segundo")

    # Mostrar estadísticas
    print("\n--- ESTADÍSTICAS DE LA TABLA HASH ---")
    stats = ht.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Probar búsquedas
    print("\n--- PROBANDO BÚSQUEDAS ---")

    # Buscar 5 documentos aleatorios
    import random

    documentos_prueba = random.sample(documentos, 5)

    inicio = time.time()
    encontrados = 0

    for doc in documentos_prueba:
        resultado = ht.search(doc["_id"])
        if resultado is not None:
            encontrados += 1
            print(f"  ✓ Encontrado: {doc['_id'][:20]}... → {resultado['title'][:40]}...")

    tiempo_busqueda = time.time() - inicio
    print(f"\n✓ {encontrados}/5 documentos encontrados")
    print(f"  Tiempo total de búsqueda: {tiempo_busqueda:.4f} segundos")
    print(f"  Tiempo promedio por búsqueda: {tiempo_busqueda/5:.4f} segundos")

    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 60)


if __name__ == "__main__":
    probar_tabla_hash()
