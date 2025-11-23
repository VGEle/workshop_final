"""
MapReduce Cost Analyzer
Unidad 4: MapReduce - Análisis de Costos

Analiza los costos de operaciones MapReduce:
- Communication Cost (transferencia de datos)
- Computation Cost (CPU)
- Storage Cost (disco)
- Wall-Clock Time (tiempo total)

Aplicación: Optimizar jobs de MapReduce
"""

import time
import sys
from mapreduce_framework import MapReduceFramework


class MapReduceCostAnalyzer:
    """
    Analizador de costos para jobs de MapReduce.
    
    Calcula múltiples métricas de costo para optimización.
    """
    
    def __init__(self):
        """Inicializa el analizador de costos."""
        self.mr = MapReduceFramework(num_workers=4)
        self.metrics = {}
    
    def analyze_job(self, mapper_func, reducer_func, data, job_name):
        """
        Analiza el costo de un job MapReduce.
        
        Parámetros:
        -----------
        mapper_func : Callable
            Función mapper
        reducer_func : Callable
            Función reducer
        data : list
            Datos de entrada
        job_name : str
            Nombre del job
            
        Retorna:
        --------
        dict
            Métricas de costo
        """
        print(f"\n{'='*60}")
        print(f"ANÁLISIS DE COSTOS: {job_name}")
        print(f"{'='*60}")
        
        # Métricas de entrada
        input_size = len(data)
        input_bytes = sys.getsizeof(data)
        
        print(f"\nInput:")
        print(f"  Registros: {input_size:,}")
        print(f"  Tamaño: {input_bytes:,} bytes ({input_bytes/1024:.2f} KB)")
        
        # Ejecutar Map Phase
        start_map = time.time()
        pairs = self.mr.map_phase(mapper_func, data)
        map_time = time.time() - start_map
        
        # Métricas de Map
        map_output_size = len(pairs)
        map_output_bytes = sys.getsizeof(pairs)
        
        print(f"\nMap Output:")
        print(f"  Pares: {map_output_size:,}")
        print(f"  Tamaño: {map_output_bytes:,} bytes ({map_output_bytes/1024:.2f} KB)")
        print(f"  Tiempo: {map_time:.3f}s")
        print(f"  Expansión: {map_output_size/input_size:.2f}x")
        
        # Ejecutar Shuffle Phase
        start_shuffle = time.time()
        grouped = self.mr.shuffle_phase(pairs)
        shuffle_time = time.time() - start_shuffle
        
        # Métricas de Shuffle
        num_keys = len(grouped)
        shuffle_bytes = sys.getsizeof(grouped)
        
        print(f"\nShuffle Output:")
        print(f"  Claves únicas: {num_keys:,}")
        print(f"  Tamaño: {shuffle_bytes:,} bytes ({shuffle_bytes/1024:.2f} KB)")
        print(f"  Tiempo: {shuffle_time:.3f}s")
        print(f"  Reducción: {map_output_size/num_keys:.2f} valores/clave")
        
        # Ejecutar Reduce Phase
        start_reduce = time.time()
        results = self.mr.reduce_phase(reducer_func, grouped)
        reduce_time = time.time() - start_reduce
        
        # Métricas de Reduce
        output_size = len(results)
        output_bytes = sys.getsizeof(results)
        
        print(f"\nReduce Output:")
        print(f"  Resultados: {output_size:,}")
        print(f"  Tamaño: {output_bytes:,} bytes ({output_bytes/1024:.2f} KB)")
        print(f"  Tiempo: {reduce_time:.3f}s")
        print(f"  Compresión: {input_size/output_size:.2f}x")
        
        # Costos calculados
        total_time = map_time + shuffle_time + reduce_time
        
        # Communication Cost: datos transferidos entre fases
        communication_cost = map_output_bytes + shuffle_bytes
        
        # Computation Cost: operaciones realizadas
        computation_cost = input_size + map_output_size + output_size
        
        # Storage Cost: espacio máximo usado
        storage_cost = max(input_bytes, map_output_bytes, shuffle_bytes, output_bytes)
        
        print(f"\n{'='*60}")
        print("RESUMEN DE COSTOS")
        print(f"{'='*60}")
        
        print(f"\nTiempo:")
        print(f"  Map:     {map_time:.3f}s ({map_time/total_time*100:.1f}%)")
        print(f"  Shuffle: {shuffle_time:.3f}s ({shuffle_time/total_time*100:.1f}%)")
        print(f"  Reduce:  {reduce_time:.3f}s ({reduce_time/total_time*100:.1f}%)")
        print(f"  Total:   {total_time:.3f}s")
        
        print(f"\nComunicación:")
        print(f"  Datos transferidos: {communication_cost:,} bytes ({communication_cost/1024:.2f} KB)")
        print(f"  Overhead: {(communication_cost/input_bytes - 1)*100:.1f}%")
        
        print(f"\nComputación:")
        print(f"  Operaciones totales: {computation_cost:,}")
        print(f"  Throughput: {computation_cost/total_time:.0f} ops/seg")
        
        print(f"\nAlmacenamiento:")
        print(f"  Pico de memoria: {storage_cost:,} bytes ({storage_cost/1024:.2f} KB)")
        print(f"  Amplificación: {storage_cost/input_bytes:.2f}x")
        
        # Calcular eficiencia
        efficiency = self._calculate_efficiency(
            input_size, map_output_size, output_size, total_time
        )
        
        print(f"\nEficiencia:")
        print(f"  Map selectivity: {efficiency['map_selectivity']:.2%}")
        print(f"  Reduce selectivity: {efficiency['reduce_selectivity']:.2%}")
        print(f"  Overall throughput: {efficiency['throughput']:.0f} rec/seg")
        print(f"  Score de eficiencia: {efficiency['score']:.1f}/100")
        
        # Guardar métricas
        self.metrics[job_name] = {
            'input_size': input_size,
            'output_size': output_size,
            'map_time': map_time,
            'shuffle_time': shuffle_time,
            'reduce_time': reduce_time,
            'total_time': total_time,
            'communication_cost': communication_cost,
            'computation_cost': computation_cost,
            'storage_cost': storage_cost,
            'efficiency': efficiency
        }
        
        return self.metrics[job_name]
    
    def _calculate_efficiency(self, input_size, map_output, reduce_output, time):
        """
        Calcula métricas de eficiencia.
        
        Parámetros:
        -----------
        input_size : int
            Tamaño de entrada
        map_output : int
            Tamaño de salida de Map
        reduce_output : int
            Tamaño de salida de Reduce
        time : float
            Tiempo total
            
        Retorna:
        --------
        dict
            Métricas de eficiencia
        """
        # Map selectivity: qué tan selectivo es el mapper
        map_selectivity = 1 - (map_output / (input_size * 10))  # Normalizado
        map_selectivity = max(0, min(1, map_selectivity))
        
        # Reduce selectivity: qué tanto reduce el reducer
        reduce_selectivity = 1 - (reduce_output / map_output) if map_output > 0 else 0
        reduce_selectivity = max(0, min(1, reduce_selectivity))
        
        # Throughput
        throughput = input_size / time if time > 0 else 0
        
        # Score general (0-100)
        score = (map_selectivity * 30 + reduce_selectivity * 30 + 
                min(throughput/1000, 1) * 40)
        
        return {
            'map_selectivity': map_selectivity,
            'reduce_selectivity': reduce_selectivity,
            'throughput': throughput,
            'score': score
        }
    
    def compare_jobs(self):
        """
        Compara múltiples jobs analizados.
        """
        if len(self.metrics) < 2:
            print("\nNecesitas analizar al menos 2 jobs para comparar")
            return
        
        print(f"\n{'='*60}")
        print("COMPARACIÓN DE JOBS")
        print(f"{'='*60}")
        
        # Tabla comparativa
        print(f"\n{'Job':<20} {'Tiempo':<10} {'Output':<10} {'Eficiencia':<12}")
        print("-" * 60)
        
        for job_name, metrics in self.metrics.items():
            time_str = f"{metrics['total_time']:.3f}s"
            output_str = f"{metrics['output_size']}"
            efficiency_str = f"{metrics['efficiency']['score']:.1f}/100"
            
            print(f"{job_name:<20} {time_str:<10} {output_str:<10} {efficiency_str:<12}")
        
        # Mejor y peor job
        best_job = max(self.metrics.items(), 
                      key=lambda x: x[1]['efficiency']['score'])
        worst_job = min(self.metrics.items(), 
                       key=lambda x: x[1]['efficiency']['score'])
        
        print(f"\nMejor job: {best_job[0]} "
              f"(eficiencia: {best_job[1]['efficiency']['score']:.1f})")
        print(f"Peor job: {worst_job[0]} "
              f"(eficiencia: {worst_job[1]['efficiency']['score']:.1f})")
        
        # Recomendaciones
        print(f"\n{'='*60}")
        print("RECOMENDACIONES DE OPTIMIZACIÓN")
        print(f"{'='*60}")
        
        for job_name, metrics in self.metrics.items():
            print(f"\n{job_name}:")
            
            eff = metrics['efficiency']
            
            if eff['map_selectivity'] < 0.3:
                print("  - Map genera demasiados pares - considerar filtrado")
            
            if eff['reduce_selectivity'] < 0.3:
                print("  - Reduce comprime poco - verificar agregación")
            
            if metrics['communication_cost'] > metrics['computation_cost']:
                print("  - Alto costo de comunicación - usar combiners")
            
            if metrics['total_time'] > 1.0:
                print("  - Considerar paralelización o particionamiento")
            
            if eff['score'] > 70:
                print("  - Job bien optimizado")


# Mappers y Reducers para pruebas
def simple_count_mapper(doc):
    """Mapper simple: cuenta por categoría"""
    category = doc.get('classificationAnalysis', {}).get('documentCategory', 'unknown')
    return [(category, 1)]


def simple_count_reducer(key, values):
    """Reducer simple: suma"""
    return (key, sum(values))


def complex_stats_mapper(doc):
    """Mapper complejo: emite múltiples pares"""
    pairs = []
    
    # Por categoría
    category = doc.get('classificationAnalysis', {}).get('documentCategory', 'unknown')
    pairs.append((f"cat:{category}", 1))
    
    # Por estado
    state = doc.get('documentState', 'unknown')
    pairs.append((f"state:{state}", 1))
    
    # Por tags
    for tag in doc.get('tags', [])[:3]:  # Solo primeros 3 tags
        pairs.append((f"tag:{tag}", 1))
    
    return pairs


def complex_stats_reducer(key, values):
    """Reducer complejo: estadísticas"""
    return (key, {'count': len(values), 'sum': sum(values)})


def filtering_mapper(doc):
    """Mapper con filtrado: solo reportes"""
    category = doc.get('classificationAnalysis', {}).get('documentCategory', 'unknown')
    
    if category == 'report':
        return [(category, 1)]
    return []


# Ejemplo de uso
if __name__ == "__main__":
    import json
    
    print("="*60)
    print("MAPREDUCE COST ANALYZER")
    print("="*60)
    
    # Cargar documentos
    print("\nCargando documentos...")
    
    with open('../data/unit4_documents.json', 'r') as f:
        documents = json.load(f)
    
    print(f"Cargados {len(documents)} documentos")
    
    # Crear analizador
    analyzer = MapReduceCostAnalyzer()
    
    # Analizar Job 1: Simple Count
    print("\n" + "="*70)
    print("JOB 1: SIMPLE COUNT")
    print("="*70)
    
    analyzer.analyze_job(
        simple_count_mapper,
        simple_count_reducer,
        documents,
        "Simple Count"
    )
    
    # Analizar Job 2: Complex Stats
    print("\n" + "="*70)
    print("JOB 2: COMPLEX STATS")
    print("="*70)
    
    analyzer.analyze_job(
        complex_stats_mapper,
        complex_stats_reducer,
        documents,
        "Complex Stats"
    )
    
    # Analizar Job 3: Filtering
    print("\n" + "="*70)
    print("JOB 3: FILTERING")
    print("="*70)
    
    analyzer.analyze_job(
        filtering_mapper,
        simple_count_reducer,
        documents,
        "Filtering"
    )
    
    # Comparar jobs
    analyzer.compare_jobs()
    
    print("\n" + "="*60)
    print("ANÁLISIS COMPLETADO")
    print("="*60)
