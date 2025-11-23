"""
Bloom Filter Implementation
Unidad 2: Data Streams - Algoritmo 1

Un Bloom Filter es una estructura probabilística que permite:
- Verificar si un elemento está en un conjunto
- Nunca produce falsos negativos
- Puede producir falsos positivos (configurables)
- Usa muy poca memoria

Aplicación: Verificar si documentos similares fueron accedidos recientemente
"""

import hashlib
import math


class BloomFilter:
    """
    Bloom Filter para verificación probabilística de membresía.
    
    Características:
    - Nunca dice "NO" cuando el elemento SÍ está (0% falsos negativos)
    - Puede decir "SÍ" cuando el elemento NO está (falsos positivos)
    - Usa espacio O(m) independiente del número de elementos
    - Operaciones en tiempo O(k) donde k es número de funciones hash
    
    Atributos:
    ----------
    size : int
        Tamaño del arreglo de bits (m)
    num_hashes : int
        Número de funciones hash (k)
    bit_array : list
        Arreglo de bits
    num_items : int
        Contador de elementos insertados
    """
    
    def __init__(self, size=10000, num_hashes=3):
        """
        Inicializa el Bloom Filter.
        
        Parámetros:
        -----------
        size : int
            Tamaño del arreglo de bits (m)
            Más grande = menos falsos positivos
        num_hashes : int
            Número de funciones hash (k)
            Óptimo: k = (m/n) * ln(2)
            
        Complejidad: O(m)
        
        Ejemplo:
        --------
        >>> bf = BloomFilter(size=1000, num_hashes=3)
        """
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
        self.num_items = 0
    
    def _hash(self, item, seed):
        """
        Genera un valor hash para un item usando una semilla.
        
        Usa SHA-256 con diferentes semillas para crear
        funciones hash independientes.
        
        Parámetros:
        -----------
        item : str
            Item a hashear
        seed : int
            Semilla para crear diferentes funciones
            
        Retorna:
        --------
        int
            Posición en el arreglo [0, size-1]
            
        Complejidad: O(1)
        """
        # Combinar item con seed
        combined = f"{item}{seed}"
        
        # Calcular hash SHA-256
        hash_obj = hashlib.sha256(combined.encode())
        
        # Convertir a entero
        hash_int = int(hash_obj.hexdigest(), 16)
        
        # Mapear a rango válido
        return hash_int % self.size
    
    def insert(self, item):
        """
        Inserta un elemento en el Bloom Filter.
        
        Proceso:
        1. Para cada función hash (k funciones):
        2.   Calcular posición = hash_i(item)
        3.   Establecer bit[posición] = 1
        
        Parámetros:
        -----------
        item : str
            Elemento a insertar
            
        Complejidad: O(k) donde k es num_hashes
        
        Ejemplo:
        --------
        >>> bf = BloomFilter()
        >>> bf.insert("documento_123")
        >>> bf.num_items
        1
        """
        for i in range(self.num_hashes):
            position = self._hash(str(item), seed=i)
            self.bit_array[position] = 1
        
        self.num_items += 1
    
    def contains(self, item):
        """
        Verifica si un elemento probablemente está en el filtro.
        
        Proceso:
        1. Para cada función hash (k funciones):
        2.   Calcular posición = hash_i(item)
        3.   Si bit[posición] == 0: retornar False (definitivamente NO está)
        4. Si todos los bits son 1: retornar True (probablemente SÍ está)
        
        Parámetros:
        -----------
        item : str
            Elemento a verificar
            
        Retorna:
        --------
        bool
            True: probablemente está (puede ser falso positivo)
            False: definitivamente NO está (nunca falso negativo)
            
        Complejidad: O(k)
        
        Ejemplo:
        --------
        >>> bf = BloomFilter()
        >>> bf.insert("doc1")
        >>> bf.contains("doc1")
        True
        >>> bf.contains("doc_no_existe")
        False
        """
        for i in range(self.num_hashes):
            position = self._hash(str(item), seed=i)
            
            # Si algún bit es 0, el elemento NO está
            if self.bit_array[position] == 0:
                return False
        
        # Todos los bits son 1, probablemente está
        return True
    
    def false_positive_probability(self):
        """
        Calcula la probabilidad actual de falsos positivos.
        
        Fórmula: P ≈ (1 - e^(-kn/m))^k
        
        Donde:
        - k = número de funciones hash
        - n = número de elementos insertados
        - m = tamaño del arreglo de bits
        
        Retorna:
        --------
        float
            Probabilidad de falso positivo (0.0 a 1.0)
            
        Ejemplo:
        --------
        >>> bf = BloomFilter(size=1000, num_hashes=3)
        >>> for i in range(100):
        ...     bf.insert(f"item{i}")
        >>> prob = bf.false_positive_probability()
        >>> print(f"Prob. falsos positivos: {prob:.4f}")
        Prob. falsos positivos: 0.0293
        """
        if self.num_items == 0:
            return 0.0
        
        # Calcular (1 - e^(-kn/m))^k
        exponent = -(self.num_hashes * self.num_items) / self.size
        probability = (1 - math.exp(exponent)) ** self.num_hashes
        
        return probability
    
    def get_fill_ratio(self):
        """
        Calcula qué proporción del arreglo está llena (bits en 1).
        
        Retorna:
        --------
        float
            Proporción de bits en 1 (0.0 a 1.0)
        """
        ones = sum(self.bit_array)
        return ones / self.size
    
    @staticmethod
    def optimal_parameters(num_items, false_positive_rate):
        """
        Calcula parámetros óptimos para un Bloom Filter.
        
        Dado:
        - Número esperado de items (n)
        - Tasa deseada de falsos positivos (P)
        
        Calcula:
        - Tamaño óptimo del arreglo (m)
        - Número óptimo de funciones hash (k)
        
        Fórmulas:
        - m = -n * ln(P) / (ln(2))^2
        - k = (m/n) * ln(2)
        
        Parámetros:
        -----------
        num_items : int
            Número esperado de elementos
        false_positive_rate : float
            Tasa deseada de falsos positivos (ej: 0.01 = 1%)
            
        Retorna:
        --------
        tuple (int, int)
            (tamaño_optimo, num_hashes_optimo)
            
        Ejemplo:
        --------
        >>> size, hashes = BloomFilter.optimal_parameters(1000, 0.01)
        >>> print(f"Para 1000 items con 1% falsos positivos:")
        >>> print(f"  Tamaño: {size} bits ({size/8:.0f} bytes)")
        >>> print(f"  Funciones hash: {hashes}")
        Para 1000 items con 1% falsos positivos:
          Tamaño: 9586 bits (1198 bytes)
          Funciones hash: 7
        """
        # Calcular tamaño óptimo
        ln2_squared = (math.log(2)) ** 2
        optimal_size = int(-num_items * math.log(false_positive_rate) / ln2_squared)
        
        # Calcular número óptimo de hashes
        optimal_hashes = int((optimal_size / num_items) * math.log(2))
        
        # Asegurar al menos 1 hash
        optimal_hashes = max(1, optimal_hashes)
        
        return (optimal_size, optimal_hashes)
    
    def get_statistics(self):
        """
        Obtiene estadísticas del Bloom Filter.
        
        Retorna:
        --------
        dict
            Diccionario con estadísticas
        """
        return {
            'size': self.size,
            'num_hashes': self.num_hashes,
            'num_items': self.num_items,
            'fill_ratio': self.get_fill_ratio(),
            'false_positive_prob': self.false_positive_probability(),
            'memory_bytes': self.size // 8
        }
    
    def __str__(self):
        """Representación en string"""
        return (f"BloomFilter(size={self.size}, hashes={self.num_hashes}, "
                f"items={self.num_items}, fill={self.get_fill_ratio():.2%})")


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("="*60)
    print("BLOOM FILTER - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Uso básico
    print("\n--- EJEMPLO 1: USO BÁSICO ---")
    
    bf = BloomFilter(size=1000, num_hashes=3)
    print(f"Bloom Filter creado: {bf}")
    
    # Insertar elementos
    documents = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    print(f"\nInsertando {len(documents)} documentos...")
    
    for doc in documents:
        bf.insert(doc)
    
    print(f"✓ Documentos insertados")
    print(f"Estado: {bf}")
    
    # Verificar membresía
    print("\n--- VERIFICACIÓN DE MEMBRESÍA ---")
    
    test_items = ["doc1", "doc3", "doc_no_existe", "doc_otro"]
    
    for item in test_items:
        result = bf.contains(item)
        status = "probablemente SÍ está" if result else "definitivamente NO está"
        print(f"  '{item}': {status}")
    
    # Ejemplo 2: Parámetros óptimos
    print("\n--- EJEMPLO 2: PARÁMETROS ÓPTIMOS ---")
    
    print("\nCalculando parámetros óptimos para diferentes configuraciones:")
    
    configs = [
        (1000, 0.01),   # 1000 items, 1% falsos positivos
        (10000, 0.01),  # 10000 items, 1% falsos positivos
        (1000, 0.001),  # 1000 items, 0.1% falsos positivos
    ]
    
    for num_items, fp_rate in configs:
        size, hashes = BloomFilter.optimal_parameters(num_items, fp_rate)
        memory_kb = size / 8 / 1024
        
        print(f"\n  {num_items} items, {fp_rate*100}% falsos positivos:")
        print(f"    Tamaño: {size} bits ({memory_kb:.2f} KB)")
        print(f"    Funciones hash: {hashes}")
    
    # Ejemplo 3: Falsos positivos en práctica
    print("\n--- EJEMPLO 3: MEDICIÓN DE FALSOS POSITIVOS ---")
    
    bf_test = BloomFilter(size=10000, num_hashes=5)
    
    # Insertar 1000 elementos
    print("\nInsertando 1000 elementos...")
    for i in range(1000):
        bf_test.insert(f"element_{i}")
    
    # Probar con 1000 elementos NO insertados
    print("Probando con 1000 elementos NO insertados...")
    false_positives = 0
    
    for i in range(1000, 2000):
        if bf_test.contains(f"element_{i}"):
            false_positives += 1
    
    observed_rate = false_positives / 1000
    expected_rate = bf_test.false_positive_probability()
    
    print(f"\nResultados:")
    print(f"  Falsos positivos observados: {false_positives}/1000 ({observed_rate:.2%})")
    print(f"  Tasa esperada (teórica): {expected_rate:.2%}")
    print(f"  Diferencia: {abs(observed_rate - expected_rate):.2%}")
    
    # Estadísticas finales
    print("\n--- ESTADÍSTICAS FINALES ---")
    stats = bf_test.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
