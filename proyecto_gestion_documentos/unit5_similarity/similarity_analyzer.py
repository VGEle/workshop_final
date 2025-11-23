"""
Similarity Analyzer - Sistema Completo
Unidad 5: Near Neighbor Search - Sistema Integrado

Sistema completo para detectar documentos similares:
1. Procesa 2000 documentos
2. Crea índice LSH
3. Detecta duplicados y documentos similares
4. Genera reportes y estadísticas

Aplicación: Detección de plagio, deduplicación, clustering
"""

import json
import time
from pathlib import Path
from collections import Counter
from jaccard_similarity import Shingling
from lsh import LSHDocumentIndex

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "unit5_documents.json"


class SimilarityAnalyzer:
    """
    Analizador completo de similitud para documentos.
    
    Pipeline completo:
    1. Cargar documentos
    2. Generar shingles
    3. Crear índice LSH
    4. Detectar similares
    5. Generar reportes
    """
    
    def __init__(self, k=3, num_hashes=100, num_bands=20):
        """
        Inicializa el analizador.
        
        Parámetros:
        -----------
        k : int
            Tamaño de shingles
        num_hashes : int
            Número de funciones hash MinHash
        num_bands : int
            Número de bandas LSH
        """
        self.shingling = Shingling(k=k, mode='char')
        self.index = LSHDocumentIndex(num_hashes=num_hashes, num_bands=num_bands)
        
        self.documents = []
        self.doc_metadata = {}
    
    def load_documents(self, documents):
        """
        Carga documentos para análisis.
        
        Parámetros:
        -----------
        documents : list
            Lista de documentos
        """
        print(f"Cargando {len(documents)} documentos...")
        
        self.documents = documents
        
        for doc in documents:
            self.doc_metadata[doc['_id']] = {
                'title': doc.get('title', 'Sin título'),
                'category': doc.get('classificationAnalysis', {}).get('documentCategory', 'unknown'),
                'tags': doc.get('tags', [])
            }
        
        print("Documentos cargados")
    
    def build_index(self):
        """
        Construye el índice LSH para todos los documentos.
        """
        print("\nConstruyendo índice LSH...")
        
        start_time = time.time()
        
        for i, doc in enumerate(self.documents):
            # Combinar título y contenido
            text = doc.get('title', '') + ' ' + doc.get('content', '')
            
            # Generar shingles
            shingles = self.shingling.get_shingles(text)
            
            # Añadir al índice
            self.index.add_document(doc['_id'], shingles)
            
            # Progreso
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(self.documents)} documentos...")
        
        elapsed = time.time() - start_time
        
        print(f"Índice construido en {elapsed:.2f}s")
        print(f"  Velocidad: {len(self.documents)/elapsed:.0f} docs/seg")
    
    def find_duplicates(self, threshold=0.9):
        """
        Encuentra documentos duplicados o casi duplicados.
        
        Parámetros:
        -----------
        threshold : float
            Umbral de similitud para considerar duplicado
            
        Retorna:
        --------
        list
            Lista de grupos de duplicados
        """
        print(f"\nBuscando duplicados (threshold: {threshold})...")
        
        start_time = time.time()
        
        # Encontrar todos los pares similares
        similar_pairs = self.index.find_all_similar_pairs(threshold=threshold)
        
        elapsed = time.time() - start_time
        
        print(f"Búsqueda completada en {elapsed:.2f}s")
        print(f"  Encontrados {len(similar_pairs)} pares similares")
        
        # Agrupar duplicados
        groups = self._group_duplicates(similar_pairs)
        
        return groups
    
    def _group_duplicates(self, pairs):
        """
        Agrupa pares en grupos de duplicados.
        
        Parámetros:
        -----------
        pairs : list
            Lista de pares similares
            
        Retorna:
        --------
        list
            Lista de grupos
        """
        # Union-Find para agrupar
        parent = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Unir pares
        for doc1, doc2, _ in pairs:
            union(doc1, doc2)
        
        # Crear grupos
        groups = {}
        for doc1, doc2, sim in pairs:
            root = find(doc1)
            if root not in groups:
                groups[root] = []
            groups[root].append((doc1, doc2, sim))
        
        return list(groups.values())
    
    def find_similar_by_category(self, threshold=0.7):
        """
        Encuentra documentos similares dentro de cada categoría.
        
        Parámetros:
        -----------
        threshold : float
            Umbral de similitud
            
        Retorna:
        --------
        dict
            Similares por categoría
        """
        print(f"\nBuscando similares por categoría (threshold: {threshold})...")
        
        # Agrupar documentos por categoría
        by_category = {}
        
        for doc_id, metadata in self.doc_metadata.items():
            category = metadata['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(doc_id)
        
        # Encontrar similares en cada categoría
        results = {}
        
        for category, doc_ids in by_category.items():
            print(f"  Procesando categoría '{category}': {len(doc_ids)} documentos...")
            
            similar_in_category = []
            
            for doc_id in doc_ids[:100]:  # Muestra de 100 por categoría
                similar = self.index.find_similar(doc_id, threshold=threshold)
                
                # Filtrar solo los de la misma categoría
                similar_same_cat = [
                    (other_id, sim) for other_id, sim in similar
                    if self.doc_metadata[other_id]['category'] == category
                ]
                
                if similar_same_cat:
                    similar_in_category.append((doc_id, similar_same_cat))
            
            results[category] = similar_in_category
        
        return results
    
    def analyze_similarity_distribution(self):
        """
        Analiza la distribución de similitudes en el corpus.
        
        Retorna:
        --------
        dict
            Estadísticas de distribución
        """
        print("\nAnalizando distribución de similitudes...")
        
        # Muestrear documentos
        sample_size = min(200, len(self.documents))
        sample_ids = [doc['_id'] for doc in self.documents[:sample_size]]
        
        similarities = []
        
        for doc_id in sample_ids:
            similar = self.index.find_similar(doc_id, threshold=0.1)
            
            for _, sim in similar:
                similarities.append(sim)
        
        if not similarities:
            return {}
        
        # Calcular estadísticas
        from statistics import mean, median, stdev
        
        return {
            'count': len(similarities),
            'mean': mean(similarities),
            'median': median(similarities),
            'stdev': stdev(similarities) if len(similarities) > 1 else 0,
            'min': min(similarities),
            'max': max(similarities)
        }
    
    def generate_report(self):
        """
        Genera reporte completo del análisis.
        """
        print("\n" + "="*70)
        print(" "*20 + "REPORTE DE SIMILITUD")
        print("="*70)
        
        print(f"\nDocumentos analizados: {len(self.documents)}")
        
        # Estadísticas del índice
        stats = self.index.lsh.get_statistics()
        
        print(f"\nConfiguración LSH:")
        print(f"  Bandas: {stats['num_bands']}")
        print(f"  Filas por banda: {stats['rows_per_band']}")
        print(f"  Threshold teórico: {stats['theoretical_threshold']:.3f}")
        print(f"  Candidatos promedio: {stats['avg_candidates_per_doc']:.1f}")
        
        # Duplicados
        print("\n--- DETECCIÓN DE DUPLICADOS (threshold: 0.9) ---")
        
        duplicates = self.find_duplicates(threshold=0.9)
        
        print(f"\nGrupos de duplicados encontrados: {len(duplicates)}")
        
        if duplicates:
            print("\nTop 5 grupos más grandes:")
            
            # Ordenar por tamaño
            duplicates.sort(key=lambda g: len(g), reverse=True)
            
            for i, group in enumerate(duplicates[:5], 1):
                print(f"\n  Grupo {i}: {len(group)} documentos similares")
                
                # Mostrar algunos documentos del grupo
                docs_in_group = set()
                for doc1, doc2, sim in group[:3]:
                    docs_in_group.add(doc1)
                    docs_in_group.add(doc2)
                
                for doc_id in list(docs_in_group)[:3]:
                    metadata = self.doc_metadata[doc_id]
                    print(f"    - {metadata['title'][:50]}... ({metadata['category']})")
        
        # Similares por categoría
        print("\n--- SIMILARES POR CATEGORÍA (threshold: 0.7) ---")
        
        by_category = self.find_similar_by_category(threshold=0.7)
        
        print("\nResumen por categoría:")
        
        for category, similar_docs in by_category.items():
            if similar_docs:
                avg_similar = sum(len(sims) for _, sims in similar_docs) / len(similar_docs)
                print(f"  {category:12}: {len(similar_docs)} docs con similares, "
                      f"promedio {avg_similar:.1f} similares/doc")
        
        # Distribución
        print("\n--- DISTRIBUCIÓN DE SIMILITUDES ---")
        
        dist = self.analyze_similarity_distribution()
        
        if dist:
            print(f"\nEstadísticas (muestra de {dist['count']} comparaciones):")
            print(f"  Media: {dist['mean']:.3f}")
            print(f"  Mediana: {dist['median']:.3f}")
            print(f"  Desv. estándar: {dist['stdev']:.3f}")
            print(f"  Rango: [{dist['min']:.3f}, {dist['max']:.3f}]")
        
        # Eficiencia
        print("\n--- ANÁLISIS DE EFICIENCIA ---")
        
        n = len(self.documents)
        total_pairs = (n * (n - 1)) // 2
        
        estimated_comparisons = n * stats['avg_candidates_per_doc']
        reduction = (1 - estimated_comparisons / total_pairs) * 100
        
        print(f"\nSin LSH:")
        print(f"  Comparaciones necesarias: {total_pairs:,}")
        
        print(f"\nCon LSH:")
        print(f"  Comparaciones estimadas: {estimated_comparisons:,.0f}")
        print(f"  Reducción: {reduction:.1f}%")
        print(f"  Aceleración: {total_pairs/estimated_comparisons:.0f}x")
        
        print("\n" + "="*70)
        print("REPORTE COMPLETADO")
        print("="*70)


# Ejecutar análisis
if __name__ == "__main__":
    print("="*60)
    print("SIMILARITY ANALYZER - SISTEMA COMPLETO")
    print("="*60)
    
    # Cargar documentos
    print("\nCargando documentos...")
    
    with DATA_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    
    print(f"Cargados {len(documents)} documentos")
    
    # Crear analizador
    analyzer = SimilarityAnalyzer(k=3, num_hashes=100, num_bands=20)
    
    # Cargar documentos
    analyzer.load_documents(documents)
    
    # Construir índice
    analyzer.build_index()
    
    # Generar reporte
    analyzer.generate_report()
    
    print("\nAnálisis completado exitosamente")
