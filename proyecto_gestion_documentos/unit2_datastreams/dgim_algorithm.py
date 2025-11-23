"""
DGIM Algorithm (Datar-Gionis-Indyk-Motwani)
Unidad 2: Data Streams - Algoritmo 5

Algoritmo para contar 1s en una ventana deslizante de bits.
Mantiene una aproximación del conteo usando espacio logarítmico.

Aplicación: Monitorear el número de búsquedas en ventanas de tiempo
"""

import time
from collections import deque
from datetime import datetime, timedelta


class Bucket:
    """
    Representa un bucket en DGIM.
    
    Un bucket almacena:
    - timestamp: momento en que terminó
    - size: tamaño del bucket (potencia de 2)
    """
    
    def __init__(self, timestamp, size):
        """
        Crea un bucket.
        
        Parámetros:
        -----------
        timestamp : int
            Marca de tiempo del último bit
        size : int
            Tamaño del bucket (número de 1s que representa)
        """
        self.timestamp = timestamp
        self.size = size
    
    def __repr__(self):
        return f"Bucket(t={self.timestamp}, size={self.size})"


class DGIM:
    """
    Implementación del algoritmo DGIM.
    
    Mantiene buckets de tamaños exponenciales para aproximar
    el conteo de 1s en una ventana deslizante.
    
    Características:
    - Espacio: O(log²N) donde N es tamaño de ventana
    - Error: ≤ 50% en el peor caso
    - Tiempo por bit: O(log N)
    
    Atributos:
    ----------
    window_size : int
        Tamaño de la ventana (N)
    buckets : deque
        Lista de buckets, ordenados por timestamp
    current_timestamp : int
        Timestamp actual
    """
    
    def __init__(self, window_size):
        """
        Inicializa DGIM.
        
        Parámetros:
        -----------
        window_size : int
            Tamaño de la ventana deslizante
            
        Ejemplo:
        --------
        >>> dgim = DGIM(window_size=100)
        """
        self.window_size = window_size
        self.buckets = deque()
        self.current_timestamp = 0
    
    def update(self, bit):
        """
        Procesa un nuevo bit del stream.
        
        Algoritmo:
        1. Incrementar timestamp
        2. Si bit == 1:
           a. Crear nuevo bucket de tamaño 1
           b. Combinar buckets si hay más de 2 del mismo tamaño
        3. Eliminar buckets fuera de ventana
        
        Parámetros:
        -----------
        bit : int
            Bit a procesar (0 o 1)
            
        Complejidad: O(log N)
        
        Ejemplo:
        --------
        >>> dgim = DGIM(window_size=10)
        >>> dgim.update(1)
        >>> dgim.update(0)
        >>> dgim.update(1)
        """
        self.current_timestamp += 1
        
        # Solo procesar 1s
        if bit == 0:
            return
        
        # Crear nuevo bucket de tamaño 1
        new_bucket = Bucket(self.current_timestamp, 1)
        self.buckets.append(new_bucket)
        
        # Combinar buckets del mismo tamaño
        self._merge_buckets()
        
        # Eliminar buckets fuera de ventana
        self._remove_expired_buckets()
    
    def _merge_buckets(self):
        """
        Combina buckets cuando hay más de 2 del mismo tamaño.
        
        Invariante: Para cada tamaño, hay a lo más 2 buckets.
        """
        # Contar buckets por tamaño
        size_counts = {}
        
        for bucket in self.buckets:
            size_counts[bucket.size] = size_counts.get(bucket.size, 0) + 1
        
        # Mientras haya más de 2 buckets del mismo tamaño
        for size in sorted(size_counts.keys()):
            while size_counts.get(size, 0) >= 3:
                # Encontrar los dos buckets más antiguos de este tamaño
                buckets_of_size = [b for b in self.buckets if b.size == size]
                
                # Tomar los dos más antiguos
                oldest = buckets_of_size[0]
                second_oldest = buckets_of_size[1]
                
                # Remover los dos buckets antiguos
                self.buckets.remove(oldest)
                self.buckets.remove(second_oldest)
                
                # Crear nuevo bucket del doble de tamaño
                # Usar timestamp del más reciente
                new_size = size * 2
                new_timestamp = second_oldest.timestamp
                merged = Bucket(new_timestamp, new_size)
                
                # Insertar en posición correcta (ordenado por timestamp)
                inserted = False
                for i, b in enumerate(self.buckets):
                    if b.timestamp > new_timestamp:
                        self.buckets.insert(i, merged)
                        inserted = True
                        break
                
                if not inserted:
                    self.buckets.append(merged)
                
                # Actualizar conteos
                size_counts[size] -= 2
                size_counts[new_size] = size_counts.get(new_size, 0) + 1
    
    def _remove_expired_buckets(self):
        """
        Elimina buckets fuera de la ventana.
        """
        cutoff_time = self.current_timestamp - self.window_size
        
        while self.buckets and self.buckets[0].timestamp <= cutoff_time:
            self.buckets.popleft()
    
    def count(self):
        """
        Cuenta aproximadamente el número de 1s en la ventana.
        
        Algoritmo:
        1. Sumar tamaños de todos los buckets excepto el más antiguo
        2. Añadir la mitad del tamaño del bucket más antiguo
        
        Retorna:
        --------
        int
            Conteo aproximado de 1s
            
        Error: ≤ 50% del valor real
        
        Ejemplo:
        --------
        >>> dgim = DGIM(window_size=10)
        >>> for bit in [1,1,0,1,0,1,1]:
        ...     dgim.update(bit)
        >>> dgim.count()  # Aproximadamente 5
        """
        if not self.buckets:
            return 0
        
        # Sumar todos excepto el primero
        total = sum(bucket.size for bucket in list(self.buckets)[1:])
        
        # Añadir mitad del primero
        total += self.buckets[0].size // 2
        
        return total
    
    def get_statistics(self):
        """
        Obtiene estadísticas del algoritmo.
        
        Retorna:
        --------
        dict
            Estadísticas
        """
        bucket_sizes = [b.size for b in self.buckets]
        
        return {
            'num_buckets': len(self.buckets),
            'window_size': self.window_size,
            'current_timestamp': self.current_timestamp,
            'bucket_sizes': bucket_sizes,
            'approximate_count': self.count()
        }
    
    def __str__(self):
        """Representación en string"""
        return f"DGIM(window={self.window_size}, buckets={len(self.buckets)}, count≈{self.count()})"


class SearchMonitor:
    """
    Monitor de búsquedas usando DGIM.
    
    Usa DGIM para rastrear búsquedas en ventanas de tiempo deslizante.
    Permite monitorear tendencias de actividad sin almacenar todo el historial.
    """
    
    def __init__(self, window_size_seconds=3600):
        """
        Inicializa el monitor.
        
        Parámetros:
        -----------
        window_size_seconds : int
            Tamaño de la ventana en segundos (default: 1 hora)
        """
        self.window_size_seconds = window_size_seconds
        self.dgim = DGIM(window_size=window_size_seconds)
        self.start_time = time.time()
    
    def record_search(self):
        """
        Registra que ocurrió una búsqueda.
        
        En el contexto de DGIM, cada búsqueda es un bit 1.
        """
        # Calular timestamp relativo en segundos
        elapsed = int(time.time() - self.start_time)
        
        # Normalizamos para que coincida con el contador del DGIM
        # Simulamos que cada llamada es un nuevo timestamp
        self.dgim.update(1)
    
    def get_search_count(self):
        """
        Obtiene el número aproximado de búsquedas en la ventana actual.
        
        Retorna:
        --------
        int
            Conteo aproximado de búsquedas
        """
        return self.dgim.count()
    
    def get_statistics(self):
        """
        Obtiene estadísticas de monitoreo.
        
        Retorna:
        --------
        dict
            Estadísticas del monitor
        """
        return {
            'approximate_searches': self.dgim.count(),
            'window_size_seconds': self.window_size_seconds,
            'num_buckets': len(self.dgim.buckets),
            'elapsed_time': int(time.time() - self.start_time)
        }


class WindowCountValidator:
    """
    Validador para comparar DGIM con conteo exacto.
    
    Útil para medir error del algoritmo DGIM.
    """
    
    def __init__(self, window_size):
        """Inicializa el validador."""
        self.window_size = window_size
        self.dgim = DGIM(window_size)
        self.bits = deque(maxlen=window_size)
    
    def update(self, bit):
        """
        Procesa un bit y actualiza ambos contadores.
        
        Parámetros:
        -----------
        bit : int
            Bit a procesar (0 o 1)
        """
        self.dgim.update(bit)
        self.bits.append(bit)
    
    def exact_count(self):
        """
        Retorna el conteo exacto de 1s en la ventana.
        
        Retorna:
        --------
        int
            Número exacto de 1s
        """
        return sum(self.bits)
    
    def approximate_count(self):
        """
        Retorna el conteo aproximado por DGIM.
        
        Retorna:
        --------
        int
            Conteo aproximado
        """
        return self.dgim.count()
    
    def error_percentage(self):
        """
        Calcula el error relativo del DGIM.
        
        Retorna:
        --------
        float
            Error porcentual
        """
        exact = self.exact_count()
        
        if exact == 0:
            return 0.0
        
        approx = self.approximate_count()
        error = abs(exact - approx) / exact
        
        return error * 100
    
    def get_comparison(self):
        """
        Compara exacto vs aproximado.
        
        Retorna:
        --------
        dict
            Comparación detallada
        """
        exact = self.exact_count()
        approx = self.approximate_count()
        
        return {
            'exact_count': exact,
            'approximate_count': approx,
            'difference': exact - approx,
            'error_percentage': self.error_percentage(),
            'num_buckets': len(self.dgim.buckets),
            'bucket_sizes': [b.size for b in self.dgim.buckets]
        }


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("="*60)
    print("DGIM ALGORITHM - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Uso básico
    print("\n--- EJEMPLO 1: USO BÁSICO ---")
    
    dgim = DGIM(window_size=20)
    
    # Stream de bits: algunos 1s, algunos 0s
    stream = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    
    print(f"Procesando stream de {len(stream)} bits...")
    print(f"Stream: {stream}")
    
    for bit in stream:
        dgim.update(bit)
    
    print(f"\n✓ Proceso completado")
    print(f"  Conteo aproximado: {dgim.count()}")
    print(f"  Número de buckets: {len(dgim.buckets)}")
    print(f"  Tamaños de buckets: {[b.size for b in dgim.buckets]}")
    
    # Ejemplo 2: Validación de exactitud
    print("\n--- EJEMPLO 2: VALIDACIÓN DE EXACTITUD ---")
    
    validator = WindowCountValidator(window_size=50)
    
    # Generar stream aleatorio
    import random
    
    print("\nProcesando 100 bits con probabilidad 40% de 1s...")
    
    for _ in range(100):
        bit = 1 if random.random() < 0.4 else 0
        validator.update(bit)
    
    print("✓ Procesado")
    
    # Comparación
    comparison = validator.get_comparison()
    
    print("\nComparación DGIM vs Exacto:")
    print(f"  Conteo exacto: {comparison['exact_count']}")
    print(f"  Conteo aproximado: {comparison['approximate_count']}")
    print(f"  Diferencia: {comparison['difference']}")
    print(f"  Error: {comparison['error_percentage']:.2f}%")
    print(f"  Buckets usados: {comparison['num_buckets']}")
    
    # Ejemplo 3: Análisis de error
    print("\n--- EJEMPLO 3: ANÁLISIS DE ERROR ---")
    
    print("\nProbando múltiples ventanas...")
    
    errors = []
    
    for window_size in [10, 20, 50, 100]:
        validator = WindowCountValidator(window_size=window_size)
        
        # Procesar 500 bits
        for _ in range(500):
            bit = 1 if random.random() < 0.5 else 0
            validator.update(bit)
        
        error = validator.error_percentage()
        errors.append((window_size, error))
    
    print("\nTamaño de ventana vs Error promedio:")
    for window_size, error in errors:
        print(f"  Ventana {window_size:3d}: {error:5.2f}% error")
    
    # Ejemplo 4: Monitor de búsquedas
    print("\n--- EJEMPLO 4: MONITOR DE BÚSQUEDAS ---")
    
    monitor = SearchMonitor(window_size_seconds=60)
    
    # Simular búsquedas
    print("\nSimulando 25 búsquedas...")
    
    for i in range(25):
        monitor.record_search()
        if (i + 1) % 5 == 0:
            print(f"  {i+1} búsquedas registradas")
    
    stats = monitor.get_statistics()
    
    print(f"\n✓ Monitoreo completado")
    print(f"  Búsquedas aproximadas: {stats['approximate_searches']}")
    print(f"  Ventana: {stats['window_size_seconds']} segundos")
    print(f"  Buckets activos: {stats['num_buckets']}")
    
    # Ejemplo 5: Demostración de espacio logarítmico
    print("\n--- EJEMPLO 5: EFICIENCIA DE ESPACIO ---")
    
    large_dgim = DGIM(window_size=1000000)
    
    print(f"\nProcesando stream con ventana de 1,000,000 bits...")
    print("Insertando 100,000 unos:")
    
    for i in range(100000):
        large_dgim.update(1)
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}: {len(large_dgim.buckets)} buckets, aprox: {large_dgim.count()}")
    
    print(f"\n✓ Completado")
    print(f"  Buckets necesarios: {len(large_dgim.buckets)}")
    print(f"  Espacio O(log²N): solo {len(large_dgim.buckets)} buckets")
    print(f"  vs {large_dgim.window_size} posiciones en método exacto")
    print(f"  Ratio de compresión: {large_dgim.window_size / len(large_dgim.buckets):.0f}x")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
