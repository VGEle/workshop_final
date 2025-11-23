"""
Document Analyzer usando MapReduce
Unidad 4: MapReduce - Aplicaciones

Análisis completo de 2000 documentos usando MapReduce:
1. Word Count
2. Document Statistics por Categoría
3. Tag Analysis
4. Processing Time Analysis
5. Join: Documents + Classification
"""

import json
import time
from pathlib import Path
from mapreduce_framework import MapReduceFramework
from collections import Counter

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "unit4_documents.json"


class DocumentAnalyzer:
    """
    Analizador de documentos usando MapReduce.
    
    Procesa 2000 documentos con múltiples análisis.
    """
    
    def __init__(self, documents):
        """
        Inicializa el analizador.
        
        Parámetros:
        -----------
        documents : list
            Lista de documentos a analizar
        """
        self.documents = documents
        self.mr = MapReduceFramework(num_workers=4)
    
    def word_count(self):
        """
        Análisis 1: Word Count
        
        Cuenta la frecuencia de palabras en todos los documentos.
        
        Retorna:
        --------
        list
            Top palabras más frecuentes
        """
        def mapper(doc):
            """Emite (palabra, 1) por cada palabra"""
            pairs = []
            
            # Procesar título
            title = doc.get('title', '').lower().split()
            for word in title:
                if len(word) > 3:  # Solo palabras de 4+ letras
                    pairs.append((word, 1))
            
            # Procesar contenido
            content = doc.get('content', '').lower().split()
            for word in content[:50]:  # Primeras 50 palabras
                if len(word) > 3:
                    pairs.append((word, 1))
            
            return pairs
        
        def reducer(key, values):
            """Suma las frecuencias"""
            return (key, sum(values))
        
        print("\n" + "="*60)
        print("ANÁLISIS 1: WORD COUNT")
        print("="*60)
        
        results = self.mr.run(mapper, reducer, self.documents)
        
        # Ordenar por frecuencia
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:20]  # Top 20
    
    def category_statistics(self):
        """
        Análisis 2: Estadísticas por Categoría
        
        Calcula estadísticas agregadas por categoría de documento.
        
        Retorna:
        --------
        list
            Estadísticas por categoría
        """
        def mapper(doc):
            """Emite (categoría, estadísticas)"""
            category = doc.get('classificationAnalysis', {}).get('documentCategory', 'unknown')
            
            # Calcular tamaño del contenido
            content_size = len(doc.get('content', ''))
            
            # Tiempo de procesamiento
            processing_time = doc.get('classificationAnalysis', {}).get('processingTime', 0)
            
            return [(category, {
                'count': 1,
                'total_size': content_size,
                'total_processing_time': processing_time,
                'has_tags': 1 if doc.get('tags') else 0
            })]
        
        def reducer(key, values):
            """Agrega estadísticas"""
            total_count = sum(v['count'] for v in values)
            total_size = sum(v['total_size'] for v in values)
            total_time = sum(v['total_processing_time'] for v in values)
            total_with_tags = sum(v['has_tags'] for v in values)
            
            return (key, {
                'count': total_count,
                'avg_size': total_size / total_count,
                'avg_processing_time': total_time / total_count,
                'percent_with_tags': (total_with_tags / total_count) * 100
            })
        
        print("\n" + "="*60)
        print("ANÁLISIS 2: ESTADÍSTICAS POR CATEGORÍA")
        print("="*60)
        
        results = self.mr.run(mapper, reducer, self.documents)
        
        return results
    
    def tag_analysis(self):
        """
        Análisis 3: Análisis de Tags
        
        Analiza la distribución y co-ocurrencia de tags.
        
        Retorna:
        --------
        dict
            Análisis de tags
        """
        def mapper(doc):
            """Emite pares (tag, 1) y co-ocurrencias"""
            tags = doc.get('tags', [])
            pairs = []
            
            # Contar tags individuales
            for tag in tags:
                pairs.append((f"tag:{tag}", 1))
            
            # Co-ocurrencias (pares de tags)
            for i in range(len(tags)):
                for j in range(i+1, len(tags)):
                    tag_pair = tuple(sorted([tags[i], tags[j]]))
                    pairs.append((f"pair:{tag_pair[0]},{tag_pair[1]}", 1))
            
            return pairs
        
        def reducer(key, values):
            """Cuenta frecuencias"""
            return (key, sum(values))
        
        print("\n" + "="*60)
        print("ANÁLISIS 3: ANÁLISIS DE TAGS")
        print("="*60)
        
        results = self.mr.run(mapper, reducer, self.documents)
        
        # Separar tags individuales y pares
        individual_tags = [(k.replace('tag:', ''), v) for k, v in results if k.startswith('tag:')]
        tag_pairs = [(k.replace('pair:', ''), v) for k, v in results if k.startswith('pair:')]
        
        # Ordenar
        individual_tags.sort(key=lambda x: x[1], reverse=True)
        tag_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'top_tags': individual_tags[:10],
            'top_pairs': tag_pairs[:10]
        }
    
    def state_transition_analysis(self):
        """
        Análisis 4: Análisis de Transiciones de Estado
        
        Analiza las transiciones entre estados de documentos.
        
        Retorna:
        --------
        list
            Transiciones más comunes
        """
        def mapper(doc):
            """Emite (transición, 1)"""
            current = doc.get('documentState', 'unknown')
            previous = doc.get('previousDocumentState', 'none')
            
            transition = f"{previous} → {current}"
            
            return [(transition, 1)]
        
        def reducer(key, values):
            """Cuenta transiciones"""
            return (key, sum(values))
        
        print("\n" + "="*60)
        print("ANÁLISIS 4: TRANSICIONES DE ESTADO")
        print("="*60)
        
        results = self.mr.run(mapper, reducer, self.documents)
        
        # Ordenar por frecuencia
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:15]
    
    def efficiency_analysis(self):
        """
        Análisis 5: Análisis de Eficiencia del Sistema
        
        Analiza la eficiencia del procesamiento por estado.
        
        Retorna:
        --------
        list
            Métricas de eficiencia por estado
        """
        def mapper(doc):
            """Emite (estado, métricas)"""
            state = doc.get('documentState', 'unknown')
            
            efficiency = doc.get('systemEfficiency', {})
            
            return [(state, {
                'throughput': efficiency.get('throughputRate', 0),
                'utilization': efficiency.get('systemUtilization', 0),
                'queue': efficiency.get('queueLength', 0),
                'bottleneck': 1 if efficiency.get('processingBottleneck') else 0,
                'count': 1
            })]
        
        def reducer(key, values):
            """Calcula promedios"""
            n = sum(v['count'] for v in values)
            
            return (key, {
                'count': n,
                'avg_throughput': sum(v['throughput'] for v in values) / n,
                'avg_utilization': sum(v['utilization'] for v in values) / n,
                'avg_queue': sum(v['queue'] for v in values) / n,
                'bottleneck_rate': sum(v['bottleneck'] for v in values) / n
            })
        
        print("\n" + "="*60)
        print("ANÁLISIS 5: EFICIENCIA DEL SISTEMA")
        print("="*60)
        
        results = self.mr.run(mapper, reducer, self.documents)
        
        return results
    
    def generate_report(self):
        """
        Genera reporte completo de todos los análisis.
        """
        print("\n" + "="*70)
        print(" "*20 + "REPORTE COMPLETO DE ANÁLISIS")
        print("="*70)
        print(f"\nTotal de documentos analizados: {len(self.documents)}")
        
        # Análisis 1: Word Count
        top_words = self.word_count()
        print("\nTop 20 palabras más frecuentes:")
        for i, (word, count) in enumerate(top_words, 1):
            bar = "█" * (count // 10)
            print(f"  {i:2d}. {word:15} {count:4d} {bar}")
        
        # Análisis 2: Categorías
        categories = self.category_statistics()
        print("\nEstadísticas por categoría:")
        for category, stats in sorted(categories):
            print(f"\n  {category}:")
            print(f"    Documentos: {stats['count']}")
            print(f"    Tamaño promedio: {stats['avg_size']:.0f} caracteres")
            print(f"    Tiempo procesamiento: {stats['avg_processing_time']:.3f}s")
            print(f"    Con tags: {stats['percent_with_tags']:.1f}%")
        
        # Análisis 3: Tags
        tag_analysis = self.tag_analysis()
        print("\nTop 10 tags más usados:")
        for i, (tag, count) in enumerate(tag_analysis['top_tags'], 1):
            print(f"  {i:2d}. {tag:15} {count:4d} documentos")
        
        print("\nTop 10 combinaciones de tags:")
        for i, (pair, count) in enumerate(tag_analysis['top_pairs'], 1):
            print(f"  {i:2d}. {pair:30} {count:3d} veces")
        
        # Análisis 4: Transiciones
        transitions = self.state_transition_analysis()
        print("\nTop 15 transiciones de estado:")
        for i, (transition, count) in enumerate(transitions, 1):
            percent = (count / len(self.documents)) * 100
            print(f"  {i:2d}. {transition:35} {count:4d} ({percent:5.2f}%)")
        
        # Análisis 5: Eficiencia
        efficiency = self.efficiency_analysis()
        print("\nEficiencia por estado:")
        for state, metrics in sorted(efficiency):
            print(f"\n  {state}:")
            print(f"    Documentos: {metrics['count']}")
            print(f"    Throughput: {metrics['avg_throughput']:.1f} docs/hora")
            print(f"    Utilización: {metrics['avg_utilization']:.1%}")
            print(f"    Cola promedio: {metrics['avg_queue']:.1f} docs")
            print(f"    Tasa de cuellos de botella: {metrics['bottleneck_rate']:.1%}")
        
        print("\n" + "="*70)
        print("✓ REPORTE COMPLETADO")
        print("="*70)


# Ejecutar análisis
if __name__ == "__main__":
    print("="*60)
    print("DOCUMENT ANALYZER - MAPREDUCE")
    print("="*60)
    
    # Cargar documentos
    print("\nCargando documentos...")
    
    with DATA_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    
    print(f"✓ Cargados {len(documents)} documentos")
    
    # Crear analizador
    analyzer = DocumentAnalyzer(documents)
    
    # Generar reporte completo
    analyzer.generate_report()
    
    print("\n✓ Análisis completado exitosamente")
