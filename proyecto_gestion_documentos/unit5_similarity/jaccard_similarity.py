"""
Jaccard Similarity and Shingling
Unidad 5: Near Neighbor Search - Fundamentos

Implementa:
1. Similitud de Jaccard entre conjuntos
2. Shingling (k-shingles) para documentos
3. Análisis de similitud entre documentos

Aplicación: Detectar documentos duplicados o muy similares
"""

from typing import Set, List, Tuple
import re
from collections import Counter


class JaccardSimilarity:
    """
    Calculadora de similitud de Jaccard.
    
    La similitud de Jaccard entre dos conjuntos A y B:
    
    J(A, B) = |A ∩ B| / |A ∪ B|
    
    Valores:
    - 0: Completamente diferentes
    - 1: Idénticos
    """
    
    @staticmethod
    def similarity(set_a: Set, set_b: Set) -> float:
        """
        Calcula similitud de Jaccard entre dos conjuntos.
        
        Parámetros:
        -----------
        set_a : Set
            Primer conjunto
        set_b : Set
            Segundo conjunto
            
        Retorna:
        --------
        float
            Similitud de Jaccard [0, 1]
            
        Ejemplo:
        --------
        >>> A = {1, 2, 3, 4}
        >>> B = {3, 4, 5, 6}
        >>> JaccardSimilarity.similarity(A, B)
        0.333  # |{3,4}| / |{1,2,3,4,5,6}| = 2/6
        """
        if len(set_a) == 0 and len(set_b) == 0:
            return 1.0
        
        if len(set_a) == 0 or len(set_b) == 0:
            return 0.0
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def distance(set_a: Set, set_b: Set) -> float:
        """
        Calcula distancia de Jaccard.
        
        Distancia = 1 - Similitud
        
        Parámetros:
        -----------
        set_a : Set
            Primer conjunto
        set_b : Set
            Segundo conjunto
            
        Retorna:
        --------
        float
            Distancia de Jaccard [0, 1]
        """
        return 1.0 - JaccardSimilarity.similarity(set_a, set_b)


class Shingling:
    """
    Generador de k-shingles (n-gramas) para documentos.
    
    Un k-shingle es una subsecuencia de k tokens consecutivos.
    
    Ejemplo:
    Texto: "the cat sat"
    2-shingles de caracteres: {"th", "he", "e ", " c", "ca", "at", ...}
    2-shingles de palabras: {"the cat", "cat sat"}
    """
    
    def __init__(self, k=3, mode='char'):
        """
        Inicializa el shingling.
        
        Parámetros:
        -----------
        k : int
            Tamaño del shingle
        mode : str
            'char' para caracteres, 'word' para palabras
            
        Ejemplo:
        --------
        >>> shingling = Shingling(k=3, mode='char')
        """
        self.k = k
        self.mode = mode
    
    def get_shingles(self, text: str) -> Set[str]:
        """
        Genera k-shingles de un texto.
        
        Parámetros:
        -----------
        text : str
            Texto a procesar
            
        Retorna:
        --------
        Set[str]
            Conjunto de k-shingles
            
        Ejemplo:
        --------
        >>> s = Shingling(k=2, mode='char')
        >>> s.get_shingles("hello")
        {'he', 'el', 'll', 'lo'}
        """
        # Normalizar texto
        text = text.lower()
        
        if self.mode == 'char':
            # Shingles de caracteres
            if len(text) < self.k:
                return {text} if text else set()
            
            shingles = set()
            for i in range(len(text) - self.k + 1):
                shingle = text[i:i + self.k]
                shingles.add(shingle)
            
            return shingles
        
        elif self.mode == 'word':
            # Shingles de palabras
            words = text.split()
            
            if len(words) < self.k:
                return {' '.join(words)} if words else set()
            
            shingles = set()
            for i in range(len(words) - self.k + 1):
                shingle = ' '.join(words[i:i + self.k])
                shingles.add(shingle)
            
            return shingles
        
        else:
            raise ValueError(f"Modo desconocido: {self.mode}")
    
    def get_shingles_with_positions(self, text: str) -> List[Tuple[str, int]]:
        """
        Genera k-shingles con sus posiciones.
        
        Útil para análisis más detallados.
        
        Parámetros:
        -----------
        text : str
            Texto a procesar
            
        Retorna:
        --------
        List[Tuple[str, int]]
            Lista de (shingle, posición)
        """
        text = text.lower()
        shingles_with_pos = []
        
        if self.mode == 'char':
            for i in range(len(text) - self.k + 1):
                shingle = text[i:i + self.k]
                shingles_with_pos.append((shingle, i))
        
        elif self.mode == 'word':
            words = text.split()
            for i in range(len(words) - self.k + 1):
                shingle = ' '.join(words[i:i + self.k])
                shingles_with_pos.append((shingle, i))
        
        return shingles_with_pos


class DocumentSimilarityAnalyzer:
    """
    Analizador de similitud entre documentos.
    
    Usa shingling y Jaccard para detectar documentos similares.
    """
    
    def __init__(self, k=3, mode='char'):
        """
        Inicializa el analizador.
        
        Parámetros:
        -----------
        k : int
            Tamaño de shingles
        mode : str
            Modo de shingling
        """
        self.shingling = Shingling(k=k, mode=mode)
        self.document_shingles = {}
    
    def add_document(self, doc_id: str, text: str):
        """
        Añade un documento al analizador.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        text : str
            Contenido del documento
        """
        shingles = self.shingling.get_shingles(text)
        self.document_shingles[doc_id] = shingles
    
    def similarity(self, doc_id1: str, doc_id2: str) -> float:
        """
        Calcula similitud entre dos documentos.
        
        Parámetros:
        -----------
        doc_id1 : str
            ID del primer documento
        doc_id2 : str
            ID del segundo documento
            
        Retorna:
        --------
        float
            Similitud de Jaccard
        """
        if doc_id1 not in self.document_shingles or doc_id2 not in self.document_shingles:
            return 0.0
        
        shingles1 = self.document_shingles[doc_id1]
        shingles2 = self.document_shingles[doc_id2]
        
        return JaccardSimilarity.similarity(shingles1, shingles2)
    
    def find_similar(self, doc_id: str, threshold=0.5, top_k=10) -> List[Tuple[str, float]]:
        """
        Encuentra documentos similares a uno dado.
        
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
        List[Tuple[str, float]]
            Lista de (doc_id, similitud)
        """
        if doc_id not in self.document_shingles:
            return []
        
        similarities = []
        
        for other_id in self.document_shingles:
            if other_id == doc_id:
                continue
            
            sim = self.similarity(doc_id, other_id)
            
            if sim >= threshold:
                similarities.append((other_id, sim))
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_all_pairs(self, threshold=0.5) -> List[Tuple[str, str, float]]:
        """
        Encuentra todos los pares de documentos similares.
        
        Parámetros:
        -----------
        threshold : float
            Similitud mínima
            
        Retorna:
        --------
        List[Tuple[str, str, float]]
            Lista de (doc_id1, doc_id2, similitud)
        """
        doc_ids = list(self.document_shingles.keys())
        pairs = []
        
        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                sim = self.similarity(doc_ids[i], doc_ids[j])
                
                if sim >= threshold:
                    pairs.append((doc_ids[i], doc_ids[j], sim))
        
        # Ordenar por similitud
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
    def get_statistics(self) -> dict:
        """
        Obtiene estadísticas del conjunto de documentos.
        
        Retorna:
        --------
        dict
            Estadísticas
        """
        num_docs = len(self.document_shingles)
        
        if num_docs == 0:
            return {}
        
        # Tamaños de conjuntos de shingles
        sizes = [len(shingles) for shingles in self.document_shingles.values()]
        
        # Shingles únicos totales
        all_shingles = set()
        for shingles in self.document_shingles.values():
            all_shingles.update(shingles)
        
        return {
            'num_documents': num_docs,
            'total_unique_shingles': len(all_shingles),
            'avg_shingles_per_doc': sum(sizes) / len(sizes),
            'min_shingles': min(sizes),
            'max_shingles': max(sizes)
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("JACCARD SIMILARITY & SHINGLING - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Similitud de Jaccard básica
    print("\n--- EJEMPLO 1: SIMILITUD DE JACCARD ---")
    
    set_a = {1, 2, 3, 4, 5}
    set_b = {4, 5, 6, 7, 8}
    
    sim = JaccardSimilarity.similarity(set_a, set_b)
    dist = JaccardSimilarity.distance(set_a, set_b)
    
    print(f"Conjunto A: {set_a}")
    print(f"Conjunto B: {set_b}")
    print(f"Intersección: {set_a & set_b}")
    print(f"Unión: {set_a | set_b}")
    print(f"Similitud de Jaccard: {sim:.3f}")
    print(f"Distancia de Jaccard: {dist:.3f}")
    
    # Ejemplo 2: Shingling de caracteres
    print("\n--- EJEMPLO 2: SHINGLING DE CARACTERES ---")
    
    shingling = Shingling(k=3, mode='char')
    
    text1 = "hello world"
    text2 = "hello word"
    
    shingles1 = shingling.get_shingles(text1)
    shingles2 = shingling.get_shingles(text2)
    
    print(f"Texto 1: '{text1}'")
    print(f"3-shingles: {sorted(shingles1)[:10]}...")
    print(f"Total: {len(shingles1)} shingles")
    
    print(f"\nTexto 2: '{text2}'")
    print(f"3-shingles: {sorted(shingles2)[:10]}...")
    print(f"Total: {len(shingles2)} shingles")
    
    sim = JaccardSimilarity.similarity(shingles1, shingles2)
    print(f"\nSimilitud entre textos: {sim:.3f}")
    
    # Ejemplo 3: Shingling de palabras
    print("\n--- EJEMPLO 3: SHINGLING DE PALABRAS ---")
    
    word_shingling = Shingling(k=2, mode='word')
    
    doc1 = "the quick brown fox jumps over the lazy dog"
    doc2 = "the quick brown dog jumps over the lazy fox"
    
    shingles1 = word_shingling.get_shingles(doc1)
    shingles2 = word_shingling.get_shingles(doc2)
    
    print(f"Doc 1: '{doc1}'")
    print(f"2-word shingles: {sorted(shingles1)}")
    
    print(f"\nDoc 2: '{doc2}'")
    print(f"2-word shingles: {sorted(shingles2)}")
    
    sim = JaccardSimilarity.similarity(shingles1, shingles2)
    print(f"\nSimilitud: {sim:.3f}")
    
    # Ejemplo 4: Análisis de documentos
    print("\n--- EJEMPLO 4: ANÁLISIS DE DOCUMENTOS ---")
    
    analyzer = DocumentSimilarityAnalyzer(k=3, mode='char')
    
    # Añadir documentos
    documents = {
        'doc1': 'Python is a programming language',
        'doc2': 'Python is a great programming language',
        'doc3': 'Java is a programming language',
        'doc4': 'Machine learning with Python',
        'doc5': 'Deep learning and neural networks'
    }
    
    print(f"Añadiendo {len(documents)} documentos...")
    
    for doc_id, text in documents.items():
        analyzer.add_document(doc_id, text)
    
    print("Documentos añadidos")
    
    # Estadísticas
    stats = analyzer.get_statistics()
    print(f"\nEstadísticas:")
    print(f"  Documentos: {stats['num_documents']}")
    print(f"  Shingles únicos totales: {stats['total_unique_shingles']}")
    print(f"  Promedio shingles/doc: {stats['avg_shingles_per_doc']:.1f}")
    
    # Encontrar similares
    print(f"\n--- DOCUMENTOS SIMILARES A 'doc1' ---")
    
    similar = analyzer.find_similar('doc1', threshold=0.3, top_k=5)
    
    print(f"Documento de referencia: '{documents['doc1']}'")
    print(f"\nDocumentos similares (umbral: 0.3):")
    
    for doc_id, sim in similar:
        print(f"  {doc_id}: {sim:.3f} - '{documents[doc_id]}'")
    
    # Todos los pares similares
    print(f"\n--- TODOS LOS PARES SIMILARES (umbral: 0.4) ---")
    
    pairs = analyzer.find_all_pairs(threshold=0.4)
    
    print(f"Encontrados {len(pairs)} pares similares:")
    
    for doc1, doc2, sim in pairs[:10]:
        print(f"\n  {doc1} <-> {doc2}: {sim:.3f}")
        print(f"    '{documents[doc1]}'")
        print(f"    '{documents[doc2]}'")
    
    print("\n" + "="*60)
    print("DEMOSTRACIÓN COMPLETADA")
    print("="*60)
