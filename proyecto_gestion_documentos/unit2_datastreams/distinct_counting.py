"""
Distinct Counting Algorithm
Unidad 2: Data Streams - Algoritmo 3

Cuenta el número exacto de elementos únicos (distintos) en un stream.
Usa un conjunto (set) para mantener elementos únicos.

Aplicación: Contar cuántos documentos únicos accedió cada usuario
"""

from collections import defaultdict


class DistinctCounter:
    """
    Contador exacto de elementos distintos.
    
    Mantiene un conjunto de elementos únicos vistos.
    Proporciona conteo exacto (no aproximado).
    
    Características:
    - Conteo exacto: sin aproximaciones
    - Espacio O(d) donde d es número de elementos distintos
    - Tiempo O(1) por inserción (promedio)
    
    Atributos:
    ----------
    distinct_set : set
        Conjunto de elementos únicos
    total_count : int
        Número total de elementos procesados (incluye duplicados)
    """
    
    def __init__(self):
        """
        Inicializa el contador de elementos distintos.
        
        Complejidad: O(1)
        """
        self.distinct_set = set()
        self.total_count = 0
    
    def add(self, item):
        """
        Añade un elemento al contador.
        
        Parámetros:
        -----------
        item : hashable
            Elemento a añadir (debe ser hashable)
            
        Complejidad: O(1) promedio
        
        Ejemplo:
        --------
        >>> counter = DistinctCounter()
        >>> counter.add("doc1")
        >>> counter.add("doc1")  # Duplicado
        >>> counter.add("doc2")
        >>> counter.get_distinct_count()
        2
        """
        self.distinct_set.add(item)
        self.total_count += 1
    
    def get_distinct_count(self):
        """
        Obtiene el número de elementos distintos.
        
        Retorna:
        --------
        int
            Número de elementos únicos
            
        Complejidad: O(1)
        """
        return len(self.distinct_set)
    
    def get_total_count(self):
        """
        Obtiene el número total de elementos (incluye duplicados).
        
        Retorna:
        --------
        int
            Número total de elementos procesados
        """
        return self.total_count
    
    def get_duplication_rate(self):
        """
        Calcula la tasa de duplicación.
        
        Tasa de duplicación = 1 - (distintos / total)
        
        Retorna:
        --------
        float
            Tasa de duplicación (0.0 = sin duplicados, 1.0 = todo duplicado)
        """
        if self.total_count == 0:
            return 0.0
        
        return 1.0 - (self.get_distinct_count() / self.total_count)
    
    def contains(self, item):
        """
        Verifica si un elemento fue visto.
        
        Parámetros:
        -----------
        item : hashable
            Elemento a verificar
            
        Retorna:
        --------
        bool
            True si el elemento fue visto
        """
        return item in self.distinct_set
    
    def get_statistics(self):
        """
        Obtiene estadísticas del contador.
        
        Retorna:
        --------
        dict
            Diccionario con estadísticas
        """
        return {
            'distinct_count': self.get_distinct_count(),
            'total_count': self.total_count,
            'duplication_rate': self.get_duplication_rate()
        }
    
    def reset(self):
        """Reinicia el contador."""
        self.distinct_set = set()
        self.total_count = 0
    
    def __str__(self):
        """Representación en string"""
        return (f"DistinctCounter(distinct={self.get_distinct_count()}, "
                f"total={self.total_count})")


class UserDocumentTracker:
    """
    Rastreador de documentos únicos por usuario.
    
    Mantiene contadores distintos para cada usuario,
    permitiendo saber cuántos documentos únicos ha accedido cada uno.
    """
    
    def __init__(self):
        """Inicializa el rastreador."""
        self.user_counters = defaultdict(DistinctCounter)
        self.global_counter = DistinctCounter()
    
    def record_access(self, user_id, document_id):
        """
        Registra que un usuario accedió a un documento.
        
        Parámetros:
        -----------
        user_id : str
            ID del usuario
        document_id : str
            ID del documento
            
        Ejemplo:
        --------
        >>> tracker = UserDocumentTracker()
        >>> tracker.record_access("user1", "doc1")
        >>> tracker.record_access("user1", "doc1")  # Mismo doc
        >>> tracker.record_access("user1", "doc2")
        >>> tracker.get_user_distinct_count("user1")
        2
        """
        # Registrar en contador del usuario
        self.user_counters[user_id].add(document_id)
        
        # Registrar en contador global
        self.global_counter.add(document_id)
    
    def get_user_distinct_count(self, user_id):
        """
        Obtiene el número de documentos únicos de un usuario.
        
        Parámetros:
        -----------
        user_id : str
            ID del usuario
            
        Retorna:
        --------
        int
            Número de documentos únicos accedidos
        """
        return self.user_counters[user_id].get_distinct_count()
    
    def get_user_total_accesses(self, user_id):
        """
        Obtiene el número total de accesos de un usuario.
        
        Parámetros:
        -----------
        user_id : str
            ID del usuario
            
        Retorna:
        --------
        int
            Número total de accesos (incluye repeticiones)
        """
        return self.user_counters[user_id].get_total_count()
    
    def get_user_statistics(self, user_id):
        """
        Obtiene estadísticas de un usuario.
        
        Parámetros:
        -----------
        user_id : str
            ID del usuario
            
        Retorna:
        --------
        dict
            Estadísticas del usuario
        """
        if user_id not in self.user_counters:
            return {
                'distinct_documents': 0,
                'total_accesses': 0,
                'revisit_rate': 0.0
            }
        
        counter = self.user_counters[user_id]
        
        return {
            'distinct_documents': counter.get_distinct_count(),
            'total_accesses': counter.get_total_count(),
            'revisit_rate': counter.get_duplication_rate()
        }
    
    def get_global_statistics(self):
        """
        Obtiene estadísticas globales del sistema.
        
        Retorna:
        --------
        dict
            Estadísticas globales
        """
        return {
            'total_users': len(self.user_counters),
            'distinct_documents_accessed': self.global_counter.get_distinct_count(),
            'total_accesses': self.global_counter.get_total_count(),
            'avg_documents_per_user': (
                sum(c.get_distinct_count() for c in self.user_counters.values()) / 
                len(self.user_counters) if self.user_counters else 0
            )
        }
    
    def get_top_users(self, n=10, by='distinct'):
        """
        Obtiene los usuarios más activos.
        
        Parámetros:
        -----------
        n : int
            Número de usuarios a retornar
        by : str
            'distinct' = por documentos únicos
            'total' = por accesos totales
            
        Retorna:
        --------
        list
            Lista de tuplas (user_id, count)
        """
        if by == 'distinct':
            key_func = lambda item: item[1].get_distinct_count()
        else:
            key_func = lambda item: item[1].get_total_count()
        
        sorted_users = sorted(
            self.user_counters.items(),
            key=key_func,
            reverse=True
        )
        
        return [
            (user_id, key_func((user_id, counter)))
            for user_id, counter in sorted_users[:n]
        ]


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("="*60)
    print("DISTINCT COUNTING - DEMOSTRACIÓN")
    print("="*60)
    
    # Ejemplo 1: Contador básico
    print("\n--- EJEMPLO 1: CONTADOR BÁSICO ---")
    
    counter = DistinctCounter()
    print(f"Contador creado: {counter}")
    
    # Añadir elementos con duplicados
    elements = ["A", "B", "C", "A", "B", "A", "D", "E", "A"]
    
    print(f"\nAñadiendo elementos: {elements}")
    for elem in elements:
        counter.add(elem)
    
    print(f"\n✓ Elementos procesados")
    print(f"  Total procesado: {counter.get_total_count()}")
    print(f"  Elementos distintos: {counter.get_distinct_count()}")
    print(f"  Tasa de duplicación: {counter.get_duplication_rate():.2%}")
    
    # Ejemplo 2: Rastreador de usuarios
    print("\n--- EJEMPLO 2: RASTREADOR DE USUARIOS ---")
    
    tracker = UserDocumentTracker()
    
    # Simular accesos de usuarios
    accesses = [
        ("user1", "doc1"),
        ("user1", "doc2"),
        ("user1", "doc1"),  # Revisita doc1
        ("user2", "doc1"),
        ("user2", "doc3"),
        ("user3", "doc2"),
        ("user1", "doc3"),
        ("user2", "doc1"),  # Revisita doc1
        ("user3", "doc4"),
        ("user1", "doc1"),  # Revisita doc1 otra vez
    ]
    
    print(f"\nSimulando {len(accesses)} accesos...")
    for user, doc in accesses:
        tracker.record_access(user, doc)
    
    print("✓ Accesos registrados")
    
    # Estadísticas por usuario
    print("\n--- ESTADÍSTICAS POR USUARIO ---")
    
    for user_id in ["user1", "user2", "user3"]:
        stats = tracker.get_user_statistics(user_id)
        print(f"\n{user_id}:")
        print(f"  Documentos únicos: {stats['distinct_documents']}")
        print(f"  Accesos totales: {stats['total_accesses']}")
        print(f"  Tasa de revisita: {stats['revisit_rate']:.2%}")
    
    # Estadísticas globales
    print("\n--- ESTADÍSTICAS GLOBALES ---")
    global_stats = tracker.get_global_statistics()
    
    print(f"  Total de usuarios: {global_stats['total_users']}")
    print(f"  Documentos únicos accedidos: {global_stats['distinct_documents_accessed']}")
    print(f"  Total de accesos: {global_stats['total_accesses']}")
    print(f"  Promedio docs por usuario: {global_stats['avg_documents_per_user']:.1f}")
    
    # Top usuarios
    print("\n--- TOP USUARIOS ---")
    
    print("\nPor documentos únicos:")
    top_distinct = tracker.get_top_users(n=3, by='distinct')
    for i, (user, count) in enumerate(top_distinct, 1):
        print(f"  {i}. {user}: {count} documentos únicos")
    
    print("\nPor accesos totales:")
    top_total = tracker.get_top_users(n=3, by='total')
    for i, (user, count) in enumerate(top_total, 1):
        print(f"  {i}. {user}: {count} accesos totales")
    
    # Ejemplo 3: Análisis a gran escala
    print("\n--- EJEMPLO 3: ANÁLISIS A GRAN ESCALA ---")
    
    import random
    
    large_tracker = UserDocumentTracker()
    
    # Simular 10000 accesos
    num_accesses = 10000
    num_users = 100
    num_docs = 500
    
    print(f"\nSimulando {num_accesses} accesos...")
    print(f"  Usuarios: {num_users}")
    print(f"  Documentos: {num_docs}")
    
    for _ in range(num_accesses):
        user = f"user_{random.randint(1, num_users)}"
        doc = f"doc_{random.randint(1, num_docs)}"
        large_tracker.record_access(user, doc)
    
    print("✓ Simulación completada")
    
    # Análisis
    global_stats = large_tracker.get_global_statistics()
    
    print("\nResultados:")
    print(f"  Usuarios activos: {global_stats['total_users']}")
    print(f"  Documentos accedidos: {global_stats['distinct_documents_accessed']}")
    print(f"  Total de accesos: {global_stats['total_accesses']}")
    print(f"  Promedio docs/usuario: {global_stats['avg_documents_per_user']:.1f}")
    
    # Top 5 usuarios más activos
    print("\n  Top 5 usuarios (por documentos únicos):")
    top_users = large_tracker.get_top_users(n=5, by='distinct')
    
    for i, (user, count) in enumerate(top_users, 1):
        total = large_tracker.get_user_total_accesses(user)
        print(f"    {i}. {user}: {count} únicos, {total} total")
    
    print("\n" + "="*60)
    print("✓ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
