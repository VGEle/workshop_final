"""
MapReduce Framework (Versión Simplificada)
Unidad 4: MapReduce

Implementación de MapReduce sin multiprocessing para compatibilidad.
Simula procesamiento distribuido de forma secuencial.

Componentes:
- Mapper: procesa registros y emite (key, value)
- Reducer: agrega valores por clave
- Framework: coordina Map → Shuffle → Reduce
"""

from collections import defaultdict
from typing import List, Tuple, Any, Callable
import time



class MapReduceFramework:
    """
    Framework MapReduce simplificado.
    
    Simula procesamiento distribuido de forma secuencial.
    
    Atributos:
    ----------
    num_workers : int
        Número de workers (simulado)
    """
    
    def __init__(self, num_workers=4):
        """
        Inicializa el framework.
        
        Parámetros:
        -----------
        num_workers : int
            Número de workers (solo informativo)
        """
        self.num_workers = num_workers
    
    def map_phase(self, mapper_func: Callable, data: List[Any]) -> List[Tuple]:
        """
        Ejecuta la fase Map.
        
        Parámetros:
        -----------
        mapper_func : Callable
            Función map(record) -> [(key, value), ...]
        data : List
            Lista de registros a procesar
            
        Retorna:
        --------
        List[Tuple]
            Lista de pares (key, value)
        """
        print(f"Map Phase: procesando {len(data)} registros...")
        
        start_time = time.time()
        all_pairs = []
        
        # Procesar cada registro
        for i, record in enumerate(data):
            try:
                result = mapper_func(record)
                if result:
                    all_pairs.extend(result)
            except Exception as e:
                print(f"  Error en registro {i}: {e}")
            
            # Mostrar progreso cada 500 registros
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(data)} registros...")
        
        elapsed = time.time() - start_time
        print(f"✓ Map completado: {len(all_pairs)} pares en {elapsed:.2f}s")
        
        return all_pairs
    
    def shuffle_phase(self, pairs: List[Tuple]) -> dict:
        """
        Fase Shuffle: agrupa valores por clave.
        
        Parámetros:
        -----------
        pairs : List[Tuple]
            Lista de pares (key, value)
            
        Retorna:
        --------
        dict
            Diccionario key → [values]
        """
        print(f"Shuffle Phase: agrupando {len(pairs)} pares...")
        
        start_time = time.time()
        grouped = defaultdict(list)
        
        for key, value in pairs:
            grouped[key].append(value)
        
        elapsed = time.time() - start_time
        print(f"✓ Shuffle completado: {len(grouped)} claves en {elapsed:.2f}s")
        
        return dict(grouped)
    
    def reduce_phase(self, reducer_func: Callable, grouped: dict) -> List[Tuple]:
        """
        Ejecuta la fase Reduce.
        
        Parámetros:
        -----------
        reducer_func : Callable
            Función reduce(key, values) -> (key, result)
        grouped : dict
            Diccionario key → [values]
            
        Retorna:
        --------
        List[Tuple]
            Lista de resultados (key, reduced_value)
        """
        print(f"Reduce Phase: procesando {len(grouped)} grupos...")
        
        start_time = time.time()
        results = []
        
        for key, values in grouped.items():
            try:
                result = reducer_func(key, values)
                results.append(result)
            except Exception as e:
                print(f"  Error en clave {key}: {e}")
        
        elapsed = time.time() - start_time
        print(f"✓ Reduce completado: {len(results)} resultados en {elapsed:.2f}s")
        
        return results
    
    def run(self, mapper_func: Callable, reducer_func: Callable, 
            data: List[Any]) -> List[Tuple]:
        """
        Ejecuta pipeline completo de MapReduce.
        
        Parámetros:
        -----------
        mapper_func : Callable
            Función mapper
        reducer_func : Callable
            Función reducer
        data : List
            Datos de entrada
            
        Retorna:
        --------
        List[Tuple]
            Resultados finales
        """
        print("\n" + "="*60)
        print("MAPREDUCE JOB")
        print("="*60)
        
        start_total = time.time()
        
        # Fase Map
        pairs = self.map_phase(mapper_func, data)
        
        # Fase Shuffle
        grouped = self.shuffle_phase(pairs)
        
        # Fase Reduce
        results = self.reduce_phase(reducer_func, grouped)
        
        elapsed_total = time.time() - start_total
        
        print(f"\n✓ Job completado en {elapsed_total:.2f}s")
        print("="*60)
        
        return results


def count_mapper(doc):
    """Mapper: Emite (tipo, 1) para cada documento"""
    return [(doc['type'], 1)]


def count_reducer(key, values):
    """Reducer: Suma los conteos"""
    return (key, sum(values))


def size_mapper(doc):
    """Mapper: Emite (tipo, tamaño)"""
    return [(doc['type'], doc['size'])]


def avg_reducer(key, values):
    """Reducer: Calcula promedio"""
    return (key, sum(values) / len(values))


def stats_mapper(doc):
    """Mapper: Emite (tipo, (tamaño, 1))"""
    return [(doc['type'], (doc['size'], 1))]


def stats_reducer(key, values):
    """Reducer: Calcula múltiples estadísticas"""
    sizes = [v[0] for v in values]
    counts = [v[1] for v in values]
    
    return (key, {
        'count': sum(counts),
        'total_size': sum(sizes),
        'avg_size': sum(sizes) / len(sizes),
        'min_size': min(sizes),
        'max_size': max(sizes)
    })


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("MAPREDUCE FRAMEWORK - DEMOSTRACIÓN")
    print("="*60)
    
    # Datos de ejemplo
    documents = [
        {'type': 'report', 'size': 100},
        {'type': 'memo', 'size': 50},
        {'type': 'report', 'size': 150},
        {'type': 'email', 'size': 20},
        {'type': 'report', 'size': 200},
        {'type': 'memo', 'size': 75},
        {'type': 'email', 'size': 15},
        {'type': 'presentation', 'size': 300},
    ]
    
    print(f"\nDocumentos de prueba: {len(documents)}")
    
    # Ejemplo 1: Contar documentos por tipo
    print("\n--- EJEMPLO 1: CONTADOR DE TIPOS ---")
    
    mr = MapReduceFramework(num_workers=4)
    results = mr.run(count_mapper, count_reducer, documents)
    
    print("\nResultados:")
    for doc_type, count in sorted(results):
        print(f"  {doc_type}: {count} documentos")
    
    # Ejemplo 2: Tamaño promedio por tipo
    print("\n--- EJEMPLO 2: TAMAÑO PROMEDIO ---")
    
    results = mr.run(size_mapper, avg_reducer, documents)
    
    print("\nTamaño promedio por tipo:")
    for doc_type, avg_size in sorted(results):
        print(f"  {doc_type}: {avg_size:.1f} KB")
    
    # Ejemplo 3: Estadísticas agregadas
    print("\n--- EJEMPLO 3: ESTADÍSTICAS COMPLEJAS ---")
    
    results = mr.run(stats_mapper, stats_reducer, documents)
    
    print("\nEstadísticas completas:")
    for doc_type, stats in sorted(results):
        print(f"\n  {doc_type}:")
        print(f"    Cantidad: {stats['count']}")
        print(f"    Tamaño total: {stats['total_size']} KB")
        print(f"    Tamaño promedio: {stats['avg_size']:.1f} KB")
        print(f"    Tamaño mínimo: {stats['min_size']} KB")
        print(f"    Tamaño máximo: {stats['max_size']} KB")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
