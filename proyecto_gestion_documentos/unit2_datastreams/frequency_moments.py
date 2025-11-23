"""
Frequency Moments Algorithm
Unidad 2: Data Streams - Algoritmo 4

Calcula momentos de frecuencia de elementos en un stream.
El momento k-ésimo se define como: F_k = Σ(f_i^k)
donde f_i es la frecuencia del elemento i.

Momento 0 (F0): Número de elementos distintos
Momento 1 (F1): Número total de elementos
Momento 2 (F2): Suma de cuadrados de frecuencias (útil para sorpresa/uniformidad)

Aplicación: Analizar la frecuencia de recuperación de documentos
"""

from collections import Counter
import math


class FrequencyMoments:
    """
    Calculador de momentos de frecuencia.
    
    Mantiene un contador de frecuencias de elementos
    y calcula diferentes momentos estadísticos.
    
    Atributos:
    ----------
    frequency_counter : Counter
        Contador de frecuencias de elementos
    """
    
    def __init__(self):
        """
        Inicializa el calculador de momentos.
        
        Complejidad: O(1)
        """
        self.frequency_counter = Counter()
    
    def add(self, item):
        """
        Añade un elemento al stream.
        
        Parámetros:
        -----------
        item : hashable
            Elemento a añadir
            
        Complejidad: O(1) promedio
        
        Ejemplo:
        --------
        >>> fm = FrequencyMoments()
        >>> fm.add("doc1")
        >>> fm.add("doc1")
        >>> fm.add("doc2")
        """
        self.frequency_counter[item] += 1
    
    def moment_0(self):
        """
        Calcula el momento 0 (F0).
        
        F0 = número de elementos distintos
        
        Retorna:
        --------
        int
            Número de elementos únicos
            
        Ejemplo:
        --------
        >>> fm = FrequencyMoments()
        >>> fm.add("A")
        >>> fm.add("B")
        >>> fm.add("A")
        >>> fm.moment_0()
        2
        """
        return len(self.frequency_counter)
    
    def moment_1(self):
        """
        Calcula el momento 1 (F1).
        
        F1 = Σ(f_i) = número total de elementos
        
        Retorna:
        --------
        int
            Número total de elementos (incluye repeticiones)
            
        Ejemplo:
        --------
        >>> fm = FrequencyMoments()
        >>> fm.add("A")  # freq = 1
        >>> fm.add("B")  # freq = 1
        >>> fm.add("A")  # freq de A = 2
        >>> fm.moment_1()
        3
        """
        return sum(self.frequency_counter.values())
    
    def moment_2(self):
        """
        Calcula el momento 2 (F2).
        
        F2 = Σ(f_i^2) = suma de cuadrados de frecuencias
        
        Este momento es útil para medir:
        - Uniformidad de la distribución
        - "Sorpresa" en el stream
        - Concentración de frecuencias
        
        Retorna:
        --------
        int
            Suma de cuadrados de frecuencias
            
        Interpretación:
        ---------------
        - F2 cercano a F1: distribución muy uniforme
        - F2 >> F1: distribución muy concentrada
        
        Ejemplo:
        --------
        >>> fm = FrequencyMoments()
        >>> fm.add("A")  # freq = 1
        >>> fm.add("B")  # freq = 1
        >>> fm.add("A")  # freq de A = 2
        >>> fm.moment_2()
        5  # 2^2 + 1^2 = 4 + 1 = 5
        """
        return sum(freq ** 2 for freq in self.frequency_counter.values())
    
    def moment_k(self, k):
        """
        Calcula el momento k-ésimo genérico.
        
        F_k = Σ(f_i^k)
        
        Parámetros:
        -----------
        k : int
            Orden del momento
            
        Retorna:
        --------
        float
            Momento k-ésimo
            
        Ejemplo:
        --------
        >>> fm = FrequencyMoments()
        >>> fm.add("A")
        >>> fm.add("A")
        >>> fm.add("B")
        >>> fm.moment_k(3)  # F3 = 2^3 + 1^3 = 8 + 1 = 9
        9
        """
        return sum(freq ** k for freq in self.frequency_counter.values())
    
    def get_entropy(self):
        """
        Calcula la entropía de Shannon de la distribución.
        
        H = -Σ(p_i * log2(p_i))
        donde p_i = f_i / F1
        
        La entropía mide la uniformidad:
        - H = 0: todo concentrado en un elemento
        - H = log2(n): distribución perfectamente uniforme
        
        Retorna:
        --------
        float
            Entropía en bits
        """
        if not self.frequency_counter:
            return 0.0
        
        total = self.moment_1()
        entropy = 0.0
        
        for freq in self.frequency_counter.values():
            if freq > 0:
                probability = freq / total
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def get_gini_coefficient(self):
        """
        Calcula el coeficiente de Gini.
        
        Gini = (F2 - F1) / (F1 * (F0 - 1))
        
        El coeficiente de Gini mide desigualdad:
        - Gini = 0: perfectamente uniforme
        - Gini = 1: máxima desigualdad
        
        Retorna:
        --------
        float
            Coeficiente de Gini (0.0 a 1.0)
        """
        f0 = self.moment_0()
        f1 = self.moment_1()
        f2 = self.moment_2()
        
        if f0 <= 1 or f1 == 0:
            return 0.0
        
        return (f2 - f1) / (f1 * (f0 - 1))
    
    def get_most_frequent(self, n=10):
        """
        Obtiene los elementos más frecuentes.
        
        Parámetros:
        -----------
        n : int
            Número de elementos a retornar
            
        Retorna:
        --------
        list
            Lista de tuplas (elemento, frecuencia)
        """
        return self.frequency_counter.most_common(n)
    
    def get_statistics(self):
        """
        Obtiene estadísticas completas.
        
        Retorna:
        --------
        dict
            Diccionario con estadísticas
        """
        f0 = self.moment_0()
        f1 = self.moment_1()
        f2 = self.moment_2()
        
        return {
            'moment_0_distinct': f0,
            'moment_1_total': f1,
            'moment_2_sum_squares': f2,
            'entropy': self.get_entropy(),
            'gini_coefficient': self.get_gini_coefficient(),
            'avg_frequency': f1 / f0 if f0 > 0 else 0
        }
    
    def reset(self):
        """Reinicia el calculador."""
        self.frequency_counter = Counter()
    
    def __str__(self):
        """Representación en string"""
        return (f"FrequencyMoments(distinct={self.moment_0()}, "
                f"total={self.moment_1()})")


class DocumentAccessAnalyzer:
    """
    Analizador de frecuencia de acceso a documentos.
    
    Usa momentos de frecuencia para analizar patrones de acceso.
    """
    
    def __init__(self):
        """Inicializa el analizador."""
        self.frequency_moments = FrequencyMoments()
        self.access_count = 0
    
    def record_access(self, document_id):
        """
        Registra un acceso a un documento.
        
        Parámetros:
        -----------
        document_id : str
            ID del documento accedido
        """
        self.frequency_moments.add(document_id)
        self.access_count += 1
    
    def analyze_distribution(self):
        """
        Analiza la distribución de accesos.
        
        Retorna:
        --------
        dict
            Análisis de la distribución
        """
        stats = self.frequency_moments.get_statistics()
        
        # Clasificar distribución basándose en entropía
        max_entropy = math.log2(stats['moment_0_distinct']) if stats['moment_0_distinct'] > 0 else 0
        entropy_ratio = stats['entropy'] / max_entropy if max_entropy > 0 else 0
        
        if entropy_ratio > 0.9:
            distribution_type = "Muy uniforme"
        elif entropy_ratio > 0.7:
            distribution_type = "Relativamente uniforme"
        elif entropy_ratio > 0.5:
            distribution_type = "Moderadamente concentrada"
        else:
            distribution_type = "Muy concentrada"
        
        return {
            'total_accesses': self.access_count,
            'unique_documents': stats['moment_0_distinct'],
            'avg_accesses_per_doc': stats['avg_frequency'],
            'entropy': stats['entropy'],
            'max_entropy': max_entropy,
            'entropy_ratio': entropy_ratio,
            'gini_coefficient': stats['gini_coefficient'],
            'distribution_type': distribution_type,
            'moment_2': stats['moment_2_sum_squares']
        }
    
    def get_popularity_report(self, top_n=10):
        """
        Genera un reporte de popularidad.
        
        Parámetros:
        -----------
        top_n : int
            Número de documentos más populares
            
        Retorna:
        --------
        dict
            Reporte de popularidad
        """
        most_frequent = self.frequency_moments.get_most_frequent(top_n)
        
        return {
            'top_documents': most_frequent,
            'total_accesses': self.access_count,
            'coverage': sum(freq for _, freq in most_frequent) / self.access_count if self.access_count > 0 else 0
        }


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("="*60)
    print("FREQUENCY MOMENTS - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Cálculo básico de momentos
    print("\n--- EJEMPLO 1: CÁLCULO BÁSICO ---")
    
    fm = FrequencyMoments()
    
    # Stream: A, B, A, C, A, B, D
    elements = ["A", "B", "A", "C", "A", "B", "D"]
    
    print(f"Stream: {elements}")
    print(f"Frecuencias: A=3, B=2, C=1, D=1")
    
    for elem in elements:
        fm.add(elem)
    
    print(f"\nMomentos:")
    print(f"  F0 (distintos): {fm.moment_0()}")
    print(f"  F1 (total): {fm.moment_1()}")
    print(f"  F2 (suma cuadrados): {fm.moment_2()}")
    print(f"    Cálculo: 3² + 2² + 1² + 1² = 9 + 4 + 1 + 1 = {fm.moment_2()}")
    
    # Ejemplo 2: Comparación de distribuciones
    print("\n--- EJEMPLO 2: COMPARACIÓN DE DISTRIBUCIONES ---")
    
    # Distribución uniforme
    print("\nDistribución UNIFORME:")
    fm_uniform = FrequencyMoments()
    for i in range(10):
        for j in range(5):  # Cada elemento aparece 5 veces
            fm_uniform.add(f"elem_{i}")
    
    stats_uniform = fm_uniform.get_statistics()
    print(f"  F0: {stats_uniform['moment_0_distinct']}")
    print(f"  F1: {stats_uniform['moment_1_total']}")
    print(f"  F2: {stats_uniform['moment_2_sum_squares']}")
    print(f"  Entropía: {stats_uniform['entropy']:.2f} bits")
    print(f"  Gini: {stats_uniform['gini_coefficient']:.3f}")
    
    # Distribución concentrada
    print("\nDistribución CONCENTRADA:")
    fm_concentrated = FrequencyMoments()
    for _ in range(40):
        fm_concentrated.add("popular")
    for i in range(9):
        fm_concentrated.add(f"rare_{i}")
    
    stats_concentrated = fm_concentrated.get_statistics()
    print(f"  F0: {stats_concentrated['moment_0_distinct']}")
    print(f"  F1: {stats_concentrated['moment_1_total']}")
    print(f"  F2: {stats_concentrated['moment_2_sum_squares']}")
    print(f"  Entropía: {stats_concentrated['entropy']:.2f} bits")
    print(f"  Gini: {stats_concentrated['gini_coefficient']:.3f}")
    
    # Ejemplo 3: Análisis de acceso a documentos
    print("\n--- EJEMPLO 3: ANÁLISIS DE ACCESOS ---")
    
    analyzer = DocumentAccessAnalyzer()
    
    # Simular accesos (siguiendo ley de potencia)
    import random
    
    documents = [f"doc_{i}" for i in range(100)]
    
    # Zipf distribution: algunos documentos muy populares
    print("\nSimulando 1000 accesos con distribución Zipf...")
    
    for _ in range(1000):
        # Probabilidad decreciente: doc_0 muy popular, doc_99 poco popular
        rank = random.choices(
            range(len(documents)),
            weights=[1/(i+1) for i in range(len(documents))]
        )[0]
        analyzer.record_access(documents[rank])
    
    print("✓ Accesos simulados")
    
    # Análisis
    print("\n--- ANÁLISIS DE DISTRIBUCIÓN ---")
    analysis = analyzer.analyze_distribution()
    
    print(f"  Total de accesos: {analysis['total_accesses']}")
    print(f"  Documentos únicos: {analysis['unique_documents']}")
    print(f"  Accesos promedio por doc: {analysis['avg_accesses_per_doc']:.1f}")
    print(f"  Entropía: {analysis['entropy']:.2f} bits")
    print(f"  Entropía máxima: {analysis['max_entropy']:.2f} bits")
    print(f"  Ratio de entropía: {analysis['entropy_ratio']:.2%}")
    print(f"  Coeficiente de Gini: {analysis['gini_coefficient']:.3f}")
    print(f"  Tipo de distribución: {analysis['distribution_type']}")
    
    # Reporte de popularidad
    print("\n--- REPORTE DE POPULARIDAD ---")
    popularity = analyzer.get_popularity_report(top_n=10)
    
    print(f"\nTop 10 documentos más accedidos:")
    for i, (doc, freq) in enumerate(popularity['top_documents'], 1):
        percentage = (freq / popularity['total_accesses']) * 100
        print(f"  {i:2d}. {doc}: {freq:3d} accesos ({percentage:5.2f}%)")
    
    print(f"\nCobertura del Top 10: {popularity['coverage']:.2%}")
    print(f"  (El Top 10 representa el {popularity['coverage']:.1%} de todos los accesos)")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
