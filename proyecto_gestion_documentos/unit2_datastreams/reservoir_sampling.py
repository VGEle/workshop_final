"""
Reservoir Sampling Algorithm
Unidad 2: Data Streams - Algoritmo 2

Reservoir Sampling permite seleccionar una muestra aleatoria uniforme
de un stream de datos donde no conocemos el tamaño total de antemano.

Aplicación: Muestrear búsquedas de usuarios para optimizar rutas de búsqueda
"""

import random


class ReservoirSampler:
    """
    Implementación de Reservoir Sampling.
    
    Algoritmo:
    - Mantiene un "reservorio" de tamaño k
    - Cada elemento del stream tiene probabilidad k/n de estar en la muestra
    - Funciona sin conocer n (tamaño total) de antemano
    
    Características:
    - Muestreo uniforme: cada elemento tiene igual probabilidad
    - Espacio O(k) constante
    - Tiempo O(1) por elemento procesado
    
    Atributos:
    ----------
    reservoir_size : int
        Tamaño del reservorio (k)
    reservoir : list
        Muestra actual
    num_seen : int
        Número total de elementos procesados
    """
    
    def __init__(self, reservoir_size):
        """
        Inicializa el Reservoir Sampler.
        
        Parámetros:
        -----------
        reservoir_size : int
            Tamaño k del reservorio
            
        Complejidad: O(1)
        
        Ejemplo:
        --------
        >>> sampler = ReservoirSampler(reservoir_size=100)
        """
        self.reservoir_size = reservoir_size
        self.reservoir = []
        self.num_seen = 0
    
    def process(self, item):
        """
        Procesa un nuevo elemento del stream.
        
        Algoritmo de Reservoir Sampling:
        
        1. Si el reservorio no está lleno (n < k):
           - Añadir el elemento directamente
        
        2. Si el reservorio está lleno (n >= k):
           - Con probabilidad k/n, reemplazar un elemento aleatorio
           - Generar j = random(0, n)
           - Si j < k, entonces reservoir[j] = item
        
        Parámetros:
        -----------
        item : any
            Elemento a procesar
            
        Complejidad: O(1)
        
        Ejemplo:
        --------
        >>> sampler = ReservoirSampler(reservoir_size=3)
        >>> for i in range(10):
        ...     sampler.process(f"item_{i}")
        >>> len(sampler.reservoir)
        3
        """
        self.num_seen += 1
        
        # Caso 1: Reservorio no está lleno
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(item)
        
        # Caso 2: Reservorio lleno - decisión probabilística
        else:
            # Generar índice aleatorio entre 0 y num_seen-1
            j = random.randint(0, self.num_seen - 1)
            
            # Si j < reservoir_size, reemplazar
            if j < self.reservoir_size:
                self.reservoir[j] = item
    
    def get_sample(self):
        """
        Obtiene la muestra actual.
        
        Retorna:
        --------
        list
            Copia de la muestra actual
            
        Ejemplo:
        --------
        >>> sampler = ReservoirSampler(reservoir_size=5)
        >>> for i in range(100):
        ...     sampler.process(i)
        >>> sample = sampler.get_sample()
        >>> len(sample)
        5
        """
        return self.reservoir.copy()
    
    def get_statistics(self):
        """
        Obtiene estadísticas del sampler.
        
        Retorna:
        --------
        dict
            Diccionario con estadísticas
        """
        return {
            'reservoir_size': self.reservoir_size,
            'current_sample_size': len(self.reservoir),
            'total_processed': self.num_seen,
            'sampling_rate': len(self.reservoir) / self.num_seen if self.num_seen > 0 else 0
        }
    
    def reset(self):
        """Reinicia el sampler."""
        self.reservoir = []
        self.num_seen = 0
    
    def __str__(self):
        """Representación en string"""
        return (f"ReservoirSampler(size={self.reservoir_size}, "
                f"processed={self.num_seen}, "
                f"sample={len(self.reservoir)})")


class SearchBehaviorAnalyzer:
    """
    Analizador de comportamiento de búsqueda usando Reservoir Sampling.
    
    Usa reservoir sampling para mantener una muestra representativa
    de búsquedas de usuarios y analizar patrones.
    """
    
    def __init__(self, sample_size=1000):
        """
        Inicializa el analizador.
        
        Parámetros:
        -----------
        sample_size : int
            Tamaño de la muestra a mantener
        """
        self.sampler = ReservoirSampler(sample_size)
        self.query_stats = {}
    
    def record_search(self, user_id, query, timestamp, results_count):
        """
        Registra una búsqueda.
        
        Parámetros:
        -----------
        user_id : str
            ID del usuario
        query : str
            Consulta de búsqueda
        timestamp : str
            Marca de tiempo
        results_count : int
            Número de resultados encontrados
        """
        search_record = {
            'user_id': user_id,
            'query': query,
            'timestamp': timestamp,
            'results_count': results_count
        }
        
        self.sampler.process(search_record)
    
    def analyze_sample(self):
        """
        Analiza la muestra de búsquedas.
        
        Retorna:
        --------
        dict
            Análisis de la muestra
        """
        sample = self.sampler.get_sample()
        
        if not sample:
            return {}
        
        # Análisis básico
        from collections import Counter
        
        queries = [s['query'] for s in sample]
        users = [s['user_id'] for s in sample]
        results = [s['results_count'] for s in sample]
        
        query_counter = Counter(queries)
        user_counter = Counter(users)
        
        return {
            'total_sampled': len(sample),
            'unique_queries': len(set(queries)),
            'unique_users': len(set(users)),
            'most_common_queries': query_counter.most_common(5),
            'most_active_users': user_counter.most_common(5),
            'avg_results': sum(results) / len(results) if results else 0,
            'queries_with_no_results': sum(1 for r in results if r == 0)
        }
    
    def get_query_suggestions(self, top_n=10):
        """
        Obtiene sugerencias de consultas populares.
        
        Parámetros:
        -----------
        top_n : int
            Número de sugerencias
            
        Retorna:
        --------
        list
            Lista de consultas más comunes
        """
        sample = self.sampler.get_sample()
        
        if not sample:
            return []
        
        from collections import Counter
        queries = [s['query'] for s in sample]
        query_counter = Counter(queries)
        
        return [query for query, count in query_counter.most_common(top_n)]


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("="*60)
    print("RESERVOIR SAMPLING - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Muestreo básico
    print("\n--- EJEMPLO 1: MUESTREO BÁSICO ---")
    
    sampler = ReservoirSampler(reservoir_size=10)
    print(f"Sampler creado: {sampler}")
    
    # Procesar 100 elementos
    print("\nProcesando 100 elementos...")
    for i in range(100):
        sampler.process(f"element_{i}")
    
    print(f"✓ Elementos procesados: {sampler.num_seen}")
    print(f"  Tamaño de muestra: {len(sampler.reservoir)}")
    
    # Verificar distribución uniforme
    print("\n--- EJEMPLO 2: VERIFICACIÓN DE UNIFORMIDAD ---")
    
    # Contar cuántas veces aparece cada elemento en múltiples ejecuciones
    element_counts = {i: 0 for i in range(20)}
    num_trials = 10000
    
    print(f"\nRealizando {num_trials} experimentos...")
    print("(muestrear 10 elementos de 20 disponibles)")
    
    for trial in range(num_trials):
        temp_sampler = ReservoirSampler(reservoir_size=10)
        
        # Procesar elementos 0-19
        for i in range(20):
            temp_sampler.process(i)
        
        # Contar elementos en la muestra
        for item in temp_sampler.get_sample():
            element_counts[item] += 1
    
    # Análisis de uniformidad
    expected_count = num_trials * 10 / 20  # 10 de 20 = 50%
    
    print(f"\nResultados:")
    print(f"  Conteo esperado por elemento: {expected_count:.0f}")
    print(f"  Conteos observados (primeros 5 elementos):")
    
    for i in range(5):
        count = element_counts[i]
        deviation = abs(count - expected_count) / expected_count * 100
        print(f"    Elemento {i}: {count} veces ({deviation:.1f}% desviación)")
    
    # Calcular desviación promedio
    deviations = [abs(count - expected_count) / expected_count * 100 
                  for count in element_counts.values()]
    avg_deviation = sum(deviations) / len(deviations)
    
    print(f"\n  Desviación promedio: {avg_deviation:.2f}%")
    print(f"  ✓ {'Muestreo uniforme correcto' if avg_deviation < 5 else 'Revisar implementación'}")
    
    # Ejemplo 3: Análisis de búsquedas
    print("\n--- EJEMPLO 3: ANÁLISIS DE BÚSQUEDAS ---")
    
    analyzer = SearchBehaviorAnalyzer(sample_size=50)
    
    # Simular búsquedas
    import datetime
    
    searches = [
        ("user1", "python tutorial", 25),
        ("user2", "data science", 30),
        ("user1", "machine learning", 15),
        ("user3", "python tutorial", 25),
        ("user2", "web development", 20),
        ("user1", "python tutorial", 25),
        ("user4", "javascript basics", 18),
        ("user3", "data science", 30),
        ("user5", "python tutorial", 25),
        ("user2", "react tutorial", 22),
    ] * 20  # Repetir 20 veces para simular 200 búsquedas
    
    print(f"\nSimulando {len(searches)} búsquedas...")
    
    for i, (user, query, results) in enumerate(searches):
        timestamp = datetime.datetime.now().isoformat()
        analyzer.record_search(user, query, timestamp, results)
    
    print(f"✓ Búsquedas registradas")
    
    # Analizar muestra
    print("\n--- ANÁLISIS DE LA MUESTRA ---")
    analysis = analyzer.analyze_sample()
    
    print(f"  Total en muestra: {analysis['total_sampled']}")
    print(f"  Consultas únicas: {analysis['unique_queries']}")
    print(f"  Usuarios únicos: {analysis['unique_users']}")
    print(f"  Resultados promedio: {analysis['avg_results']:.1f}")
    
    print("\n  Consultas más comunes:")
    for query, count in analysis['most_common_queries']:
        print(f"    '{query}': {count} veces")
    
    print("\n  Usuarios más activos:")
    for user, count in analysis['most_active_users']:
        print(f"    {user}: {count} búsquedas")
    
    # Sugerencias
    print("\n--- SUGERENCIAS DE BÚSQUEDA ---")
    suggestions = analyzer.get_query_suggestions(top_n=5)
    
    print("  Top 5 sugerencias:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"    {i}. {suggestion}")
    
    # Estadísticas del sampler
    print("\n--- ESTADÍSTICAS DEL SAMPLER ---")
    stats = analyzer.sampler.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
