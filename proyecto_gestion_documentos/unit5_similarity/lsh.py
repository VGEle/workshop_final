"""
Locality Sensitive Hashing (LSH)
Unidad 5: Near Neighbor Search - LSH

LSH permite encontrar pares similares SIN comparar todos los pares:
- Sin LSH: O(n²) comparaciones
- Con LSH: O(n) comparaciones

Idea: "Hash" documentos similares al mismo bucket

Aplicación: Detectar documentos duplicados en 2000 documentos
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict
import time


class LSH:
    """
    Implementación de Locality Sensitive Hashing.
    
    LSH divide firmas MinHash en bandas:
    - Documentos con >=1 banda idéntica → candidatos
    - Solo comparar candidatos (no todos los pares)
    
    Parámetros clave:
    - b = número de bandas
    - r = filas por banda
    - Threshold ~ (1/b)^(1/r)
    
    Ejemplo:
    - 100 hashes, 20 bandas, 5 filas/banda
    - Threshold ~ 0.47
    
    Atributos:
    ----------
    num_bands : int
        Número de bandas
    rows_per_band : int
        Filas por banda
    """
    
    def __init__(self, num_bands=20, rows_per_band=5):
        """
        Inicializa LSH.
        
        Parámetros:
        -----------
        num_bands : int
            Número de bandas
        rows_per_band : int
            Filas por banda
            
        Ejemplo:
        --------
        >>> lsh = LSH(num_bands=20, rows_per_band=5)
        """
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.signature_length = num_bands * rows_per_band
        
        # Hash tables (uno por banda)
        self.hash_tables = [defaultdict(list) for _ in range(num_bands)]
        
        # Documentos indexados
        self.signatures = {}
    
    def _hash_band(self, band: List[int]) -> int:
        """
        Hash una banda a un bucket.
        
        Parámetros:
        -----------
        band : List[int]
            Banda (sublista de firma)
            
        Retorna:
        --------
        int
            Bucket hash
        """
        return hash(tuple(band))
    
    def index_document(self, doc_id: str, signature: List[int]):
        """
        Indexa un documento en LSH.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        signature : List[int]
            Firma MinHash del documento
            
        Ejemplo:
        --------
        >>> lsh.index_document('doc1', signature)
        """
        if len(signature) != self.signature_length:
            raise ValueError(
                f"Firma debe tener {self.signature_length} elementos, "
                f"tiene {len(signature)}"
            )
        
        # Guardar firma
        self.signatures[doc_id] = signature
        
        # Dividir en bandas e indexar
        for band_idx in range(self.num_bands):
            # Extraer banda
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            
            # Hash banda a bucket
            bucket = self._hash_band(band)
            
            # Añadir documento al bucket
            self.hash_tables[band_idx][bucket].append(doc_id)
    
    def get_candidates(self, doc_id: str) -> Set[str]:
        """
        Obtiene documentos candidatos (potencialmente similares).
        
        Candidato = comparte al menos 1 banda idéntica
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
            
        Retorna:
        --------
        Set[str]
            IDs de documentos candidatos
            
        Ejemplo:
        --------
        >>> candidates = lsh.get_candidates('doc1')
        >>> print(len(candidates))
        15
        """
        if doc_id not in self.signatures:
            return set()
        
        candidates = set()
        signature = self.signatures[doc_id]
        
        # Para cada banda
        for band_idx in range(self.num_bands):
            # Extraer banda
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            
            # Hash banda
            bucket = self._hash_band(band)
            
            # Añadir documentos del mismo bucket
            if bucket in self.hash_tables[band_idx]:
                candidates.update(self.hash_tables[band_idx][bucket])
        
        # Remover el documento mismo
        candidates.discard(doc_id)
        
        return candidates
    
    def find_similar_pairs(self, threshold=0.5) -> List[Tuple[str, str]]:
        """
        Encuentra todos los pares similares usando LSH.
        
        Proceso:
        1. Para cada documento, obtener candidatos
        2. Comparar solo con candidatos
        3. Verificar si similitud >= threshold
        
        Parámetros:
        -----------
        threshold : float
            Umbral de similitud
            
        Retorna:
        --------
        List[Tuple[str, str]]
            Pares de documentos similares
            
        Ejemplo:
        --------
        >>> pairs = lsh.find_similar_pairs(threshold=0.7)
        >>> print(f"Encontrados {len(pairs)} pares")
        """
        similar_pairs = set()
        
        for doc_id in self.signatures:
            candidates = self.get_candidates(doc_id)
            
            for candidate_id in candidates:
                # Evitar duplicados
                pair = tuple(sorted([doc_id, candidate_id]))
                
                if pair not in similar_pairs:
                    # Calcular similitud real
                    sig1 = self.signatures[doc_id]
                    sig2 = self.signatures[candidate_id]
                    
                    similarity = self._estimate_similarity(sig1, sig2)
                    
                    if similarity >= threshold:
                        similar_pairs.add(pair)
        
        return list(similar_pairs)
    
    def _estimate_similarity(self, sig1: List[int], sig2: List[int]) -> float:
        """
        Estima similitud entre dos firmas.
        
        Parámetros:
        -----------
        sig1 : List[int]
            Primera firma
        sig2 : List[int]
            Segunda firma
            
        Retorna:
        --------
        float
            Similitud estimada
        """
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas de LSH.
        
        Retorna:
        --------
        dict
            Estadísticas
        """
        # Tamaños de buckets
        bucket_sizes = []
        for hash_table in self.hash_tables:
            bucket_sizes.extend([len(docs) for docs in hash_table.values()])
        
        # Candidatos promedio
        avg_candidates = 0
        if self.signatures:
            total_candidates = sum(
                len(self.get_candidates(doc_id)) 
                for doc_id in list(self.signatures.keys())[:100]  # Muestra
            )
            avg_candidates = total_candidates / min(100, len(self.signatures))
        
        return {
            'num_documents': len(self.signatures),
            'num_bands': self.num_bands,
            'rows_per_band': self.rows_per_band,
            'total_buckets': sum(len(ht) for ht in self.hash_tables),
            'avg_bucket_size': sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0,
            'avg_candidates_per_doc': avg_candidates,
            'theoretical_threshold': (1/self.num_bands) ** (1/self.rows_per_band)
        }


class LSHDocumentIndex:
    """
    Índice completo combinando MinHash y LSH.
    
    Pipeline completo:
    1. Documento -> Shingles
    2. Shingles -> MinHash signature
    3. Signature -> LSH buckets
    4. Búsqueda rápida de similares
    """
    
    def __init__(self, num_hashes=100, num_bands=20):
        """
        Inicializa el índice.
        
        Parámetros:
        -----------
        num_hashes : int
            Número de funciones hash para MinHash
        num_bands : int
            Número de bandas para LSH
        """
        from minhash import MinHash
        
        self.minhash = MinHash(num_hashes=num_hashes)
        
        rows_per_band = num_hashes // num_bands
        self.lsh = LSH(num_bands=num_bands, rows_per_band=rows_per_band)
        
        self.documents = {}
    
    def add_document(self, doc_id: str, shingles: Set[str]):
        """
        Añade un documento al índice.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        shingles : Set[str]
            Shingles del documento
        """
        # Crear firma MinHash
        signature = self.minhash.create_signature(shingles)
        
        # Indexar en LSH
        self.lsh.index_document(doc_id, signature)
        
        # Guardar shingles
        self.documents[doc_id] = shingles
    
    def find_similar(self, doc_id: str, threshold=0.5) -> List[Tuple[str, float]]:
        """
        Encuentra documentos similares a uno dado.
        
        Usa LSH para obtener candidatos rápidamente.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        threshold : float
            Umbral de similitud
            
        Retorna:
        --------
        List[Tuple[str, float]]
            Lista de (doc_id, similitud)
        """
        # Obtener candidatos con LSH
        candidates = self.lsh.get_candidates(doc_id)
        
        # Calcular similitud con candidatos
        results = []
        
        for candidate_id in candidates:
            sig1 = self.lsh.signatures[doc_id]
            sig2 = self.lsh.signatures[candidate_id]
            
            similarity = self.lsh._estimate_similarity(sig1, sig2)
            
            if similarity >= threshold:
                results.append((candidate_id, similarity))
        
        # Ordenar por similitud
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def find_all_similar_pairs(self, threshold=0.5) -> List[Tuple[str, str, float]]:
        """
        Encuentra todos los pares similares.
        
        Parámetros:
        -----------
        threshold : float
            Umbral de similitud
            
        Retorna:
        --------
        List[Tuple[str, str, float]]
            Lista de (doc_id1, doc_id2, similitud)
        """
        pairs = self.lsh.find_similar_pairs(threshold=threshold)
        
        # Añadir similitudes
        pairs_with_sim = []
        
        for doc_id1, doc_id2 in pairs:
            sig1 = self.lsh.signatures[doc_id1]
            sig2 = self.lsh.signatures[doc_id2]
            
            similarity = self.lsh._estimate_similarity(sig1, sig2)
            pairs_with_sim.append((doc_id1, doc_id2, similarity))
        
        # Ordenar por similitud
        pairs_with_sim.sort(key=lambda x: x[2], reverse=True)
        
        return pairs_with_sim


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("LSH (LOCALITY SENSITIVE HASHING) - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: LSH básico
    print("\n--- EJEMPLO 1: LSH BÁSICO ---")
    
    from minhash import MinHash
    
    mh = MinHash(num_hashes=100)
    lsh = LSH(num_bands=20, rows_per_band=5)
    
    # Crear documentos de prueba
    docs = {
        'doc1': {'apple', 'banana', 'orange', 'grape', 'melon'},
        'doc2': {'apple', 'banana', 'orange', 'grape', 'kiwi'},
        'doc3': {'apple', 'banana', 'mango', 'papaya', 'kiwi'},
        'doc4': {'carrot', 'potato', 'onion', 'tomato', 'pepper'},
        'doc5': {'carrot', 'potato', 'onion', 'tomato', 'lettuce'}
    }
    
    print(f"Indexando {len(docs)} documentos...")
    
    for doc_id, elements in docs.items():
        signature = mh.create_signature(elements)
        lsh.index_document(doc_id, signature)
    
    print("Documentos indexados")
    
    # Estadísticas
    stats = lsh.get_statistics()
    print(f"\nEstadísticas LSH:")
    print(f"  Documentos: {stats['num_documents']}")
    print(f"  Bandas: {stats['num_bands']}")
    print(f"  Filas por banda: {stats['rows_per_band']}")
    print(f"  Buckets totales: {stats['total_buckets']}")
    print(f"  Candidatos promedio: {stats['avg_candidates_per_doc']:.1f}")
    print(f"  Threshold teórico: {stats['theoretical_threshold']:.3f}")
    
    # Buscar candidatos
    print(f"\n--- CANDIDATOS PARA 'doc1' ---")
    
    candidates = lsh.get_candidates('doc1')
    print(f"Candidatos encontrados: {candidates}")
    
    # Encontrar pares similares
    print(f"\n--- PARES SIMILARES (threshold: 0.5) ---")
    
    pairs = lsh.find_similar_pairs(threshold=0.5)
    print(f"Encontrados {len(pairs)} pares similares:")
    
    for doc_id1, doc_id2 in pairs:
        sig1 = lsh.signatures[doc_id1]
        sig2 = lsh.signatures[doc_id2]
        sim = lsh._estimate_similarity(sig1, sig2)
        print(f"  {doc_id1} <-> {doc_id2}: {sim:.3f}")
    
    # Ejemplo 2: Índice completo
    print("\n--- EJEMPLO 2: ÍNDICE COMPLETO ---")
    
    from jaccard_similarity import Shingling
    
    index = LSHDocumentIndex(num_hashes=100, num_bands=20)
    shingling = Shingling(k=3, mode='char')
    
    # Documentos de texto
    text_docs = {
        'doc1': 'Python is a programming language',
        'doc2': 'Python is a great programming language',
        'doc3': 'Java is a programming language',
        'doc4': 'Machine learning with Python',
        'doc5': 'Deep learning uses neural networks',
        'doc6': 'Python programming for beginners',
        'doc7': 'Advanced Java programming techniques',
        'doc8': 'Neural networks and deep learning'
    }
    
    print(f"Indexando {len(text_docs)} documentos de texto...")
    
    start_time = time.time()
    
    for doc_id, text in text_docs.items():
        shingles = shingling.get_shingles(text)
        index.add_document(doc_id, shingles)
    
    indexing_time = time.time() - start_time
    
    print(f"Indexados en {indexing_time:.3f}s")
    
    # Buscar similares
    print(f"\n--- SIMILARES A 'doc1' (threshold: 0.4) ---")
    print(f"Documento: '{text_docs['doc1']}'")
    
    start_time = time.time()
    similar = index.find_similar('doc1', threshold=0.4)
    search_time = time.time() - start_time
    
    print(f"\nEncontrados {len(similar)} similares en {search_time*1000:.1f}ms:")
    
    for doc_id, sim in similar:
        print(f"  {doc_id}: {sim:.3f} - '{text_docs[doc_id]}'")
    
    # Todos los pares
    print(f"\n--- TODOS LOS PARES SIMILARES (threshold: 0.5) ---")
    
    start_time = time.time()
    all_pairs = index.find_all_similar_pairs(threshold=0.5)
    pairs_time = time.time() - start_time
    
    print(f"Encontrados {len(all_pairs)} pares en {pairs_time*1000:.1f}ms:")
    
    for doc_id1, doc_id2, sim in all_pairs[:10]:
        print(f"\n  {doc_id1} <-> {doc_id2}: {sim:.3f}")
        print(f"    '{text_docs[doc_id1]}'")
        print(f"    '{text_docs[doc_id2]}'")
    
    # Ejemplo 3: Análisis de eficiencia
    print("\n--- EJEMPLO 3: ANÁLISIS DE EFICIENCIA ---")
    
    n = len(text_docs)
    total_pairs = (n * (n - 1)) // 2
    
    print(f"\nDocumentos: {n}")
    print(f"Pares totales posibles: {total_pairs}")
    print(f"Pares candidatos revisados: ~{len(all_pairs)}")
    print(f"Reducción: {(1 - len(all_pairs)/total_pairs)*100:.1f}%")
    
    print(f"\nComparaciones:")
    print(f"  Sin LSH: {total_pairs} comparaciones")
    print(f"  Con LSH: ~{n * stats['avg_candidates_per_doc']:.0f} comparaciones")
    print(f"  Aceleración: {total_pairs / (n * stats['avg_candidates_per_doc']):.1f}x")
    
    print("\n" + "="*60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("="*60)
