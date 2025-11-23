"""
MinHash Signatures
Unidad 5: Near Neighbor Search - MinHash

MinHash permite estimar similitud de Jaccard usando firmas compactas:
- Conjunto original: miles de elementos
- Firma MinHash: ~100 números
- Preserva similitud de Jaccard

Teorema: P(h_min(A) = h_min(B)) = J(A, B)

Aplicación: Comparar documentos eficientemente
"""

import random
import hashlib
from typing import Set, List


class MinHash:
    """
    Implementación de MinHash.
    
    MinHash crea firmas compactas que preservan similitud de Jaccard.
    
    Proceso:
    1. Definir n funciones hash
    2. Para cada conjunto S y cada función h:
       signature[i] = min(h(x) para x en S)
    3. Estimar similitud comparando firmas
    
    Atributos:
    ----------
    num_hashes : int
        Número de funciones hash (tamaño de firma)
    seed : int
        Semilla para reproducibilidad
    """
    
    def __init__(self, num_hashes=100, seed=42):
        """
        Inicializa MinHash.
        
        Parámetros:
        -----------
        num_hashes : int
            Número de funciones hash
            Mayor número = mejor precisión, más lento
        seed : int
            Semilla aleatoria
            
        Ejemplo:
        --------
        >>> mh = MinHash(num_hashes=100)
        """
        self.num_hashes = num_hashes
        self.seed = seed
        
        # Generar parámetros para funciones hash
        random.seed(seed)
        self.hash_functions = []
        
        for i in range(num_hashes):
            # Parámetros para hash: (a, b) en h(x) = (a*x + b) mod prime
            a = random.randint(1, 2**32 - 1)
            b = random.randint(0, 2**32 - 1)
            self.hash_functions.append((a, b))
    
    def _hash(self, element: str) -> int:
        """
        Convierte un elemento a número usando hash.
        
        Parámetros:
        -----------
        element : str
            Elemento a hashear
            
        Retorna:
        --------
        int
            Valor hash
        """
        return int(hashlib.md5(element.encode()).hexdigest(), 16)
    
    def create_signature(self, elements: Set[str]) -> List[int]:
        """
        Crea firma MinHash para un conjunto.
        
        Parámetros:
        -----------
        elements : Set[str]
            Conjunto de elementos
            
        Retorna:
        --------
        List[int]
            Firma MinHash
            
        Ejemplo:
        --------
        >>> mh = MinHash(num_hashes=10)
        >>> signature = mh.create_signature({'a', 'b', 'c'})
        >>> len(signature)
        10
        """
        if not elements:
            return [float('inf')] * self.num_hashes
        
        signature = [float('inf')] * self.num_hashes
        
        # Para cada elemento del conjunto
        for element in elements:
            # Hash base del elemento
            element_hash = self._hash(element)
            
            # Para cada función hash
            for i, (a, b) in enumerate(self.hash_functions):
                # Calcular hash con función i
                h = (a * element_hash + b) % (2**32 - 1)
                
                # Mantener el mínimo
                signature[i] = min(signature[i], h)
        
        return signature
    
    def estimate_similarity(self, signature1: List[int], signature2: List[int]) -> float:
        """
        Estima similitud de Jaccard desde firmas.
        
        Similitud ≈ (número de posiciones iguales) / (total de posiciones)
        
        Parámetros:
        -----------
        signature1 : List[int]
            Primera firma
        signature2 : List[int]
            Segunda firma
            
        Retorna:
        --------
        float
            Similitud estimada [0, 1]
            
        Ejemplo:
        --------
        >>> sig1 = [1, 5, 3, 9]
        >>> sig2 = [1, 7, 3, 2]
        >>> mh.estimate_similarity(sig1, sig2)
        0.5  # 2 de 4 posiciones iguales
        """
        if len(signature1) != len(signature2):
            raise ValueError("Las firmas deben tener el mismo tamaño")
        
        matches = sum(1 for a, b in zip(signature1, signature2) if a == b)
        
        return matches / len(signature1)


class MinHashDocumentIndex:
    """
    Índice de documentos usando MinHash.
    
    Permite búsqueda eficiente de documentos similares.
    """
    
    def __init__(self, num_hashes=100):
        """
        Inicializa el índice.
        
        Parámetros:
        -----------
        num_hashes : int
            Número de funciones hash
        """
        self.minhash = MinHash(num_hashes=num_hashes)
        self.signatures = {}
        self.documents = {}
    
    def add_document(self, doc_id: str, shingles: Set[str]):
        """
        Añade un documento al índice.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        shingles : Set[str]
            Conjunto de shingles del documento
        """
        signature = self.minhash.create_signature(shingles)
        self.signatures[doc_id] = signature
        self.documents[doc_id] = shingles
    
    def similarity(self, doc_id1: str, doc_id2: str) -> float:
        """
        Estima similitud entre dos documentos.
        
        Parámetros:
        -----------
        doc_id1 : str
            ID del primer documento
        doc_id2 : str
            ID del segundo documento
            
        Retorna:
        --------
        float
            Similitud estimada
        """
        if doc_id1 not in self.signatures or doc_id2 not in self.signatures:
            return 0.0
        
        sig1 = self.signatures[doc_id1]
        sig2 = self.signatures[doc_id2]
        
        return self.minhash.estimate_similarity(sig1, sig2)
    
    def find_similar(self, doc_id: str, threshold=0.5, top_k=10) -> List[tuple]:
        """
        Encuentra documentos similares.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento de referencia
        threshold : float
            Similitud mínima
        top_k : int
            Número máximo de resultados
            
        Retorna:
        --------
        List[tuple]
            Lista de (doc_id, similitud_estimada)
        """
        if doc_id not in self.signatures:
            return []
        
        similarities = []
        
        for other_id in self.signatures:
            if other_id == doc_id:
                continue
            
            sim = self.similarity(doc_id, other_id)
            
            if sim >= threshold:
                similarities.append((other_id, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del índice.
        
        Retorna:
        --------
        dict
            Estadísticas
        """
        if not self.signatures:
            return {}
        
        # Tamaño de firmas
        signature_size = len(next(iter(self.signatures.values())))
        
        # Espacio usado
        num_docs = len(self.signatures)
        space_signatures = num_docs * signature_size * 8  # 8 bytes por int
        
        # Espacio original
        total_shingles = sum(len(s) for s in self.documents.values())
        space_original = total_shingles * 50  # ~50 bytes por shingle (estimado)
        
        compression = space_original / space_signatures if space_signatures > 0 else 0
        
        return {
            'num_documents': num_docs,
            'signature_size': signature_size,
            'space_signatures_kb': space_signatures / 1024,
            'space_original_kb': space_original / 1024,
            'compression_ratio': compression
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("MINHASH - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: MinHash básico
    print("\n--- EJEMPLO 1: MINHASH BÁSICO ---")
    
    mh = MinHash(num_hashes=10)
    
    set_a = {'apple', 'banana', 'orange', 'grape'}
    set_b = {'apple', 'banana', 'kiwi', 'melon'}
    
    sig_a = mh.create_signature(set_a)
    sig_b = mh.create_signature(set_b)
    
    print(f"Conjunto A: {set_a}")
    print(f"Conjunto B: {set_b}")
    print(f"\nFirma A (primeros 5): {sig_a[:5]}")
    print(f"Firma B (primeros 5): {sig_b[:5]}")
    
    # Similitud real
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    real_jaccard = intersection / union
    
    # Similitud estimada
    estimated_jaccard = mh.estimate_similarity(sig_a, sig_b)
    
    print(f"\nSimilitud de Jaccard real: {real_jaccard:.3f}")
    print(f"Similitud de Jaccard estimada: {estimated_jaccard:.3f}")
    print(f"Error: {abs(real_jaccard - estimated_jaccard):.3f}")
    
    # Ejemplo 2: Precisión con diferentes tamaños
    print("\n--- EJEMPLO 2: ANÁLISIS DE PRECISIÓN ---")
    
    set1 = set(f"element_{i}" for i in range(100))
    set2 = set(f"element_{i}" for i in range(50, 150))
    
    # Jaccard real
    real_sim = len(set1 & set2) / len(set1 | set2)
    
    print(f"Conjuntos de 100 elementos con 50 en común")
    print(f"Similitud real: {real_sim:.3f}")
    print(f"\nProbando diferentes tamaños de firma:")
    
    for num_hashes in [10, 50, 100, 200]:
        mh_test = MinHash(num_hashes=num_hashes)
        sig1 = mh_test.create_signature(set1)
        sig2 = mh_test.create_signature(set2)
        est_sim = mh_test.estimate_similarity(sig1, sig2)
        error = abs(real_sim - est_sim)
        
        print(f"  {num_hashes:3d} hashes: estimado = {est_sim:.3f}, error = {error:.3f}")
    
    # Ejemplo 3: Índice de documentos
    print("\n--- EJEMPLO 3: ÍNDICE DE DOCUMENTOS ---")
    
    from jaccard_similarity import Shingling
    
    index = MinHashDocumentIndex(num_hashes=100)
    shingling = Shingling(k=3, mode='char')
    
    # Documentos de prueba
    documents = {
        'doc1': 'Python is a high-level programming language',
        'doc2': 'Python is a great high-level programming language',
        'doc3': 'Java is a high-level programming language',
        'doc4': 'Machine learning with Python and TensorFlow',
        'doc5': 'Deep learning uses neural networks'
    }
    
    print(f"Añadiendo {len(documents)} documentos al índice...")
    
    for doc_id, text in documents.items():
        shingles = shingling.get_shingles(text)
        index.add_document(doc_id, shingles)
    
    print("Documentos indexados")
    
    # Estadísticas
    stats = index.get_statistics()
    print(f"\nEstadísticas del índice:")
    print(f"  Documentos: {stats['num_documents']}")
    print(f"  Tamaño de firma: {stats['signature_size']}")
    print(f"  Espacio firmas: {stats['space_signatures_kb']:.2f} KB")
    print(f"  Espacio original: {stats['space_original_kb']:.2f} KB")
    print(f"  Ratio de compresión: {stats['compression_ratio']:.1f}x")
    
    # Buscar similares
    print(f"\n--- DOCUMENTOS SIMILARES A 'doc1' ---")
    print(f"Documento: '{documents['doc1']}'")
    
    similar = index.find_similar('doc1', threshold=0.4, top_k=5)
    
    print(f"\nDocumentos similares (umbral: 0.4):")
    for doc_id, sim in similar:
        print(f"  {doc_id}: {sim:.3f} - '{documents[doc_id]}'")
    
    # Ejemplo 4: Comparación exacta vs estimada
    print("\n--- EJEMPLO 4: COMPARACIÓN EXACTA VS ESTIMADA ---")
    
    from jaccard_similarity import JaccardSimilarity
    
    print(f"\n{'Par':<15} {'Exacta':<10} {'Estimada':<10} {'Error':<10}")
    print("-" * 50)
    
    pairs = [
        ('doc1', 'doc2'),
        ('doc1', 'doc3'),
        ('doc1', 'doc4'),
        ('doc2', 'doc3'),
    ]
    
    for doc_id1, doc_id2 in pairs:
        # Similitud exacta
        shingles1 = shingling.get_shingles(documents[doc_id1])
        shingles2 = shingling.get_shingles(documents[doc_id2])
        exact = JaccardSimilarity.similarity(shingles1, shingles2)
        
        # Similitud estimada
        estimated = index.similarity(doc_id1, doc_id2)
        
        error = abs(exact - estimated)
        
        pair_str = f"{doc_id1}-{doc_id2}"
        print(f"{pair_str:<15} {exact:<10.3f} {estimated:<10.3f} {error:<10.3f}")
    
    print("\n" + "="*60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("="*60)
