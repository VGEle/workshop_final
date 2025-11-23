"""
Implementación de Tabla Hash para el Sistema de Gestión de Documentos
Unidad 1: Hashing y Algoritmos Aleatorizados
"""


class HashTable:
    """
    Tabla Hash para almacenamiento eficiente de documentos.

    Usa encadenamiento (chaining) para resolver colisiones.
    Cada posición del arreglo contiene una lista de documentos.

    Atributos:
    ----------
    size : int
        Tamaño de la tabla hash
    table : list
        Arreglo que contiene las listas de documentos
    num_items : int
        Número total de documentos almacenados
    """

    def __init__(self, size=1000):
        """
        Inicializa la tabla hash con un tamaño específico.

        Parámetros:
        -----------
        size : int
            Tamaño inicial de la tabla (default: 1000)

        Complejidad Temporal: O(n) donde n es el tamaño
        Complejidad Espacial: O(n)
        """
        self.size = size
        self.table = [[] for _ in range(size)]
        self.num_items = 0

    def _hash_function(self, key):
        """
        Función hash que convierte una clave en un índice.

        Usa el método de división: hash(key) = key % size

        Parámetros:
        -----------
        key : str
            Clave a hashear (típicamente el _id del documento)

        Retorna:
        --------
        int
            Índice en el rango [0, size-1]

        Complejidad Temporal: O(len(key))
        """
        # Convertir string a número usando suma de códigos ASCII
        hash_value = 0
        for char in key:
            hash_value += ord(char)

        # Aplicar módulo para obtener índice válido
        return hash_value % self.size

    def insert(self, document):
        """
        Inserta un documento en la tabla hash.

        Parámetros:
        -----------
        document : dict
            Documento con al menos el campo '_id'

        Retorna:
        --------
        bool
            True si la inserción fue exitosa, False si ya existía

        Complejidad Temporal:
        - Promedio: O(1)
        - Peor caso: O(n) si hay muchas colisiones

        Ejemplo:
        --------
        >>> ht = HashTable()
        >>> doc = {'_id': '123', 'title': 'Test'}
        >>> ht.insert(doc)
        True
        """
        # Obtener clave del documento
        key = document["_id"]

        # Calcular índice usando función hash
        index = self._hash_function(key)

        # Verificar si el documento ya existe en la lista
        for existing_doc in self.table[index]:
            if existing_doc["_id"] == key:
                # Documento ya existe, no insertamos duplicado
                return False

        # Añadir documento a la lista en esa posición
        self.table[index].append(document)
        self.num_items += 1

        return True

    def search(self, key):
        """
        Busca un documento por su clave (_id).

        Parámetros:
        -----------
        key : str
            Clave del documento a buscar

        Retorna:
        --------
        dict or None
            Documento si se encuentra, None si no existe

        Complejidad Temporal:
        - Promedio: O(1)
        - Peor caso: O(n) si hay muchas colisiones

        Ejemplo:
        --------
        >>> ht = HashTable()
        >>> doc = {'_id': '123', 'title': 'Test'}
        >>> ht.insert(doc)
        >>> result = ht.search('123')
        >>> print(result['title'])
        Test
        """
        # Calcular índice
        index = self._hash_function(key)

        # Buscar en la lista de esa posición
        for document in self.table[index]:
            if document["_id"] == key:
                return document

        # No encontrado
        return None

    def delete(self, key):
        """
        Elimina un documento de la tabla hash.

        Parámetros:
        -----------
        key : str
            Clave del documento a eliminar

        Retorna:
        --------
        bool
            True si se eliminó, False si no existía

        Complejidad Temporal:
        - Promedio: O(1)
        - Peor caso: O(n)
        """
        # Calcular índice
        index = self._hash_function(key)

        # Buscar y eliminar de la lista
        for i, document in enumerate(self.table[index]):
            if document["_id"] == key:
                self.table[index].pop(i)
                self.num_items -= 1
                return True

        return False

    def get_load_factor(self):
        """
        Calcula el factor de carga de la tabla hash.

        Factor de carga = número de elementos / tamaño de la tabla
        Un factor alto (>0.7) indica que hay muchas colisiones.

        Retorna:
        --------
        float
            Factor de carga

        Ejemplo:
        --------
        >>> ht = HashTable(size=100)
        >>> # Insertar 50 documentos
        >>> ht.get_load_factor()
        0.5
        """
        return self.num_items / self.size

    def get_statistics(self):
        """
        Obtiene estadísticas sobre la distribución de la tabla hash.

        Retorna:
        --------
        dict
            Diccionario con estadísticas:
            - total_items: Número total de documentos
            - table_size: Tamaño de la tabla
            - load_factor: Factor de carga
            - empty_buckets: Posiciones vacías
            - max_chain_length: Longitud de la cadena más larga
            - avg_chain_length: Longitud promedio de cadenas no vacías
        """
        # Contar buckets vacíos
        empty_buckets = sum(1 for bucket in self.table if len(bucket) == 0)

        # Encontrar longitud máxima de cadena
        max_chain = max(len(bucket) for bucket in self.table)

        # Calcular longitud promedio de cadenas no vacías
        non_empty_buckets = [len(bucket) for bucket in self.table if len(bucket) > 0]
        avg_chain = sum(non_empty_buckets) / len(non_empty_buckets) if non_empty_buckets else 0

        return {
            "total_items": self.num_items,
            "table_size": self.size,
            "load_factor": self.get_load_factor(),
            "empty_buckets": empty_buckets,
            "max_chain_length": max_chain,
            "avg_chain_length": round(avg_chain, 2),
        }

    def __len__(self):
        """Retorna el número de documentos en la tabla"""
        return self.num_items

    def __str__(self):
        """Representación en string de la tabla"""
        stats = self.get_statistics()
        return f"HashTable(size={self.size}, items={self.num_items}, load_factor={stats['load_factor']:.2f})"


# Ejemplo de uso
if __name__ == "__main__":
    # Crear tabla hash
    ht = HashTable(size=100)

    # Insertar algunos documentos
    docs = [
        {"_id": "doc1", "title": "Primer Documento"},
        {"_id": "doc2", "title": "Segundo Documento"},
        {"_id": "doc3", "title": "Tercer Documento"},
    ]

    for doc in docs:
        ht.insert(doc)

    print("Tabla Hash creada:")
    print(ht)
    print("\nEstadísticas:")
    for key, value in ht.get_statistics().items():
        print(f"  {key}: {value}")

    # Buscar un documento
    result = ht.search("doc2")
    print(f"\nBúsqueda de 'doc2':")
    print(f"  Resultado: {result['title']}")
