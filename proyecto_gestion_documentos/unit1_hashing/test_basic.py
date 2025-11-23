"""
Test básico para verificar que la tabla hash funciona
"""

import sys
import os

# Añadir el directorio padre al path para poder importar
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unit1_hashing.hash_table import HashTable


def test_crear_tabla_hash():
    """Test 1: Crear una tabla hash vacía"""
    print("\n=== Test 1: Crear Tabla Hash ===")

    ht = HashTable(size=100)

    print(f"Tamaño de la tabla: {ht.size}")
    print(f"Número de items: {ht.num_items}")
    print(f"¿La tabla está vacía? {len(ht) == 0}")

    assert ht.size == 100, "El tamaño debería ser 100"
    assert ht.num_items == 0, "Debería estar vacía"

    print("✓ Test 1 pasado")


def test_insertar_documento():
    """Test 2: Insertar un documento"""
    print("\n=== Test 2: Insertar Documento ===")

    ht = HashTable(size=100)

    doc = {
        "_id": "test123",
        "title": "Documento de Prueba",
        "content": "Este es un documento de prueba",
    }

    resultado = ht.insert(doc)

    print(f"¿Inserción exitosa? {resultado}")
    print(f"Número de items ahora: {ht.num_items}")

    assert resultado is True, "La inserción debería ser exitosa"
    assert ht.num_items == 1, "Debería haber 1 documento"

    print("✓ Test 2 pasado")


def test_buscar_documento():
    """Test 3: Buscar un documento"""
    print("\n=== Test 3: Buscar Documento ===")

    ht = HashTable(size=100)

    # Insertar documento
    doc = {
        "_id": "buscar123",
        "title": "Documento a Buscar",
        "content": "Contenido del documento",
    }
    ht.insert(doc)

    # Buscar el documento
    resultado = ht.search("buscar123")

    print(f"Documento encontrado: {resultado is not None}")
    if resultado:
        print(f"Título: {resultado['title']}")

    assert resultado is not None, "El documento debería encontrarse"
    assert resultado["title"] == "Documento a Buscar", "El título debería coincidir"

    print("✓ Test 3 pasado")


def test_buscar_documento_inexistente():
    """Test 4: Buscar un documento que no existe"""
    print("\n=== Test 4: Buscar Documento Inexistente ===")

    ht = HashTable(size=100)

    # Buscar documento que no existe
    resultado = ht.search("noexiste999")

    print(f"¿Documento encontrado? {resultado is not None}")

    assert resultado is None, "No debería encontrar el documento"

    print("✓ Test 4 pasado")


def test_insertar_multiples_documentos():
    """Test 5: Insertar múltiples documentos"""
    print("\n=== Test 5: Insertar Múltiples Documentos ===")

    ht = HashTable(size=100)

    # Insertar 10 documentos
    for i in range(10):
        doc = {
            "_id": f"doc{i}",
            "title": f"Documento {i}",
            "content": f"Contenido del documento {i}",
        }
        ht.insert(doc)

    print(f"Documentos insertados: {ht.num_items}")

    # Buscar algunos documentos
    doc5 = ht.search("doc5")
    doc9 = ht.search("doc9")

    print(f"¿Se encontró doc5? {doc5 is not None}")
    print(f"¿Se encontró doc9? {doc9 is not None}")

    assert ht.num_items == 10, "Deberían haber 10 documentos"
    assert doc5 is not None, "doc5 debería existir"
    assert doc9 is not None, "doc9 debería existir"

    print("✓ Test 5 pasado")


def test_estadisticas():
    """Test 6: Verificar estadísticas de la tabla"""
    print("\n=== Test 6: Estadísticas ===")

    ht = HashTable(size=100)

    # Insertar 20 documentos
    for i in range(20):
        doc = {"_id": f"stat{i}", "title": f"Doc {i}"}
        ht.insert(doc)

    stats = ht.get_statistics()

    print(f"Total de items: {stats['total_items']}")
    print(f"Tamaño de tabla: {stats['table_size']}")
    print(f"Factor de carga: {stats['load_factor']:.2f}")
    print(f"Buckets vacíos: {stats['empty_buckets']}")
    print(f"Longitud máxima de cadena: {stats['max_chain_length']}")
    print(f"Longitud promedio de cadena: {stats['avg_chain_length']}")

    assert stats["total_items"] == 20, "Deberían ser 20 items"
    assert stats["load_factor"] == 0.2, "Factor de carga debería ser 0.2"

    print("✓ Test 6 pasado")


# Ejecutar todos los tests
if __name__ == "__main__":
    print("=" * 60)
    print("EJECUTANDO TESTS DE LA TABLA HASH")
    print("=" * 60)

    try:
        test_crear_tabla_hash()
        test_insertar_documento()
        test_buscar_documento()
        test_buscar_documento_inexistente()
        test_insertar_multiples_documentos()
        test_estadisticas()

        print("\n" + "=" * 60)
        print("✓ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ TEST FALLÓ: {e}")
    except Exception as e:
        print(f"\n✗ ERROR INESPERADO: {e}")
