"""
Document Workflow Analyzer
Unidad 3: Cadenas de Markov - Sistema Integrado

Sistema completo que integra:
- Cadenas de Markov para flujo de documentos
- PageRank para importancia
- Random Walk para navegación

Aplicación: Análisis completo del ciclo de vida de documentos
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from markov_chain import MarkovChain
from pagerank import PageRank
from random_walk import RandomWalk

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "unit3_documents.json"


class DocumentWorkflowAnalyzer:
    """
    Analizador completo de flujo de trabajo de documentos.
    
    Combina múltiples técnicas de cadenas de Markov para
    analizar y optimizar el flujo de documentos en un sistema.
    """
    
    def __init__(self):
        """Inicializa el analizador."""
        self.markov_chain = None
        self.pagerank = PageRank(damping_factor=0.85)
        self.random_walk = RandomWalk()
        
        self.documents = []
        self.state_transitions = defaultdict(lambda: defaultdict(int))
        self.document_network = defaultdict(list)
    
    def load_documents(self, documents):
        """
        Carga documentos para análisis.
        
        Parámetros:
        -----------
        documents : list
            Lista de documentos con campos de Unidad 3
        """
        self.documents = documents
        print(f"✓ Cargados {len(documents)} documentos")
    
    def build_state_transition_model(self):
        """
        Construye modelo de transición de estados.
        
        Analiza las transiciones de estado de los documentos
        para crear la matriz de transición.
        """
        print("\nConstruyendo modelo de transición de estados...")
        
        # Recolectar transiciones
        states = set()
        
        for doc in self.documents:
            current_state = doc.get('documentState')
            previous_state = doc.get('previousDocumentState')
            
            if current_state:
                states.add(current_state)
            if previous_state:
                states.add(previous_state)
                # Registrar transición
                self.state_transitions[previous_state][current_state] += 1
        
        states = sorted(list(states))
        
        # Crear cadena de Markov
        self.markov_chain = MarkovChain(states)
        
        # Construir matriz de transición
        transition_matrix = []
        
        for from_state in states:
            row = []
            total_transitions = sum(self.state_transitions[from_state].values())
            
            for to_state in states:
                if total_transitions > 0:
                    prob = self.state_transitions[from_state][to_state] / total_transitions
                else:
                    # Sin datos, usar uniforme
                    prob = 1.0 / len(states)
                
                row.append(prob)
            
            transition_matrix.append(row)
        
        self.markov_chain.set_transition_matrix(transition_matrix)
        
        print(f"✓ Modelo construido con {len(states)} estados")
        
        return states
    
    def build_document_network(self):
        """
        Construye red de documentos basada en similitud.
        
        Usa tags para crear conexiones entre documentos similares.
        """
        print("\nConstruyendo red de documentos...")
        
        # Crear índice de tags
        tag_to_docs = defaultdict(list)
        
        for doc in self.documents:
            doc_id = doc['_id']
            tags = doc.get('tags', [])
            
            for tag in tags:
                tag_to_docs[tag].append(doc_id)
        
        # Crear conexiones basadas en tags compartidos
        connections = 0
        
        for doc in self.documents:
            doc_id = doc['_id']
            tags = doc.get('tags', [])
            
            # Encontrar documentos similares
            similar_docs = set()
            for tag in tags:
                similar_docs.update(tag_to_docs[tag])
            
            # Remover a sí mismo
            similar_docs.discard(doc_id)
            
            # Añadir conexiones
            for similar_id in similar_docs:
                self.pagerank.add_edge(doc_id, similar_id)
                self.random_walk.add_edge(doc_id, similar_id)
                self.document_network[doc_id].append(similar_id)
                connections += 1
        
        print(f"✓ Red construida: {connections} conexiones")
    
    def analyze_workflow(self):
        """
        Analiza el flujo de trabajo de documentos.
        
        Retorna:
        --------
        dict
            Análisis completo del workflow
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE FLUJO DE TRABAJO")
        print("="*60)
        
        # 1. Distribución estacionaria
        stationary = self.markov_chain.stationary_distribution()
        
        print("\n--- DISTRIBUCIÓN ESTACIONARIA ---")
        print("(Porcentaje de documentos en cada estado a largo plazo)")
        
        for i, state in enumerate(self.markov_chain.states):
            prob = stationary[i]
            bar = "|" * int(prob * 100)
            print(f"  {state:12}: {prob:.4f} ({prob*100:5.2f}%) {bar}")
        
        # 2. Cuellos de botella
        print("\n--- DETECCIÓN DE CUELLOS DE BOTELLA ---")
        
        bottlenecks = []
        for doc in self.documents:
            if doc.get('systemEfficiency', {}).get('processingBottleneck'):
                bottlenecks.append(doc)
        
        bottleneck_rate = len(bottlenecks) / len(self.documents) * 100
        print(f"  Documentos con cuello de botella: {len(bottlenecks)} ({bottleneck_rate:.1f}%)")
        
        # Estados con mayor tiempo de permanencia
        print("\n  Estados con mayor tiempo de permanencia:")
        state_times = defaultdict(list)
        
        for doc in self.documents:
            state = doc.get('documentState')
            time = doc.get('timeInCurrentState', 0)
            if state:
                state_times[state].append(time)
        
        for state in sorted(state_times.keys(), 
                           key=lambda s: np.mean(state_times[s]), 
                           reverse=True)[:3]:
            avg_time = np.mean(state_times[state])
            print(f"    {state:12}: {avg_time:.1f} minutos promedio")
        
        # 3. Tiempos de procesamiento
        print("\n--- TIEMPOS ESPERADOS DE PROCESAMIENTO ---")
        
        for target_state in ['archived', 'retrieved']:
            if target_state in self.markov_chain.states:
                time = self.markov_chain.expected_time_to_state(target_state)
                print(f"  Hasta '{target_state}': ~{time:.1f} transiciones")
        
        return {
            'stationary_distribution': dict(zip(self.markov_chain.states, stationary)),
            'bottleneck_rate': bottleneck_rate,
            'avg_state_times': {state: np.mean(times) for state, times in state_times.items()}
        }
    
    def analyze_document_importance(self):
        """
        Analiza la importancia de documentos usando PageRank.
        
        Retorna:
        --------
        dict
            Análisis de importancia
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE IMPORTANCIA (PAGERANK)")
        print("="*60)
        
        # Calcular PageRank
        scores = self.pagerank.compute_pagerank()
        
        # Top documentos
        top_docs = self.pagerank.get_top_nodes(n=10)
        
        print("\nTop 10 documentos más importantes:")
        
        for i, (doc_id, score) in enumerate(top_docs, 1):
            # Buscar info del documento
            doc_info = next((d for d in self.documents if d['_id'] == doc_id), None)
            
            if doc_info:
                title = doc_info.get('title', 'Sin título')[:40]
                category = doc_info.get('classificationAnalysis', {}).get('documentCategory', 'N/A')
                
                bar = "|" * int(score * 200)
                print(f"{i:2d}. [{score:.6f}] {bar}")
                print(f"    ID: {doc_id[:20]}...")
                print(f"    Título: {title}")
                print(f"    Categoría: {category}")
        
        # Estadísticas
        stats = self.pagerank.get_statistics()
        
        print(f"\nEstadísticas de la red:")
        print(f"  Nodos: {stats['num_nodes']}")
        print(f"  Conexiones: {stats['num_edges']}")
        print(f"  PageRank promedio: {stats['avg_pagerank']:.6f}")
        print(f"  PageRank máximo: {stats['max_pagerank']:.6f}")
        
        return {
            'top_documents': top_docs,
            'statistics': stats
        }
    
    def analyze_navigation_patterns(self, num_simulations=100):
        """
        Analiza patrones de navegación usando Random Walk.
        
        Parámetros:
        -----------
        num_simulations : int
            Número de simulaciones
            
        Retorna:
        --------
        dict
            Análisis de navegación
        """
        print("\n" + "="*60)
        print("ANÁLISIS DE PATRONES DE NAVEGACIÓN")
        print("="*60)
        
        # Seleccionar documentos iniciales populares
        start_docs = [doc['_id'] for doc in self.documents[:10]]
        
        all_visits = Counter()
        avg_coverage = []
        
        print(f"\nSimulando {num_simulations} sesiones de navegación...")
        
        for _ in range(num_simulations):
            start = np.random.choice(start_docs)
            
            # Simular navegación
            path = self.random_walk.simulate_user_navigation(
                start,
                num_clicks=30,
                return_probability=0.1
            )
            
            # Registrar visitas
            all_visits.update(path)
            
            # Calcular cobertura
            coverage = len(set(path)) / len(self.random_walk.nodes)
            avg_coverage.append(coverage)
        
        print(f"✓ Simulaciones completadas")
        
        # Análisis
        total_visits = sum(all_visits.values())
        
        print(f"\nCobertura promedio: {np.mean(avg_coverage):.1%}")
        print(f"(Porcentaje de documentos visitados por sesión)")
        
        print(f"\nTop 10 documentos más visitados:")
        for i, (doc_id, count) in enumerate(all_visits.most_common(10), 1):
            freq = count / total_visits
            
            # Buscar info
            doc_info = next((d for d in self.documents if d['_id'] == doc_id), None)
            title = doc_info.get('title', 'Sin título')[:40] if doc_info else 'Desconocido'
            
            bar = "|" * int(freq * 100)
            print(f"{i:2d}. {freq:.3f} {bar}")
            print(f"    {title}")
        
        return {
            'avg_coverage': np.mean(avg_coverage),
            'top_visited': all_visits.most_common(10)
        }
    
    def generate_recommendations(self, doc_id, method='pagerank', n=5):
        """
        Genera recomendaciones de documentos similares.
        
        Parámetros:
        -----------
        doc_id : str
            ID del documento
        method : str
            'pagerank' o 'random_walk'
        n : int
            Número de recomendaciones
            
        Retorna:
        --------
        list
            Lista de documentos recomendados
        """
        if method == 'pagerank':
            # Recomendaciones basadas en importancia
            neighbors = self.document_network.get(doc_id, [])
            
            if not neighbors:
                return []
            
            # Ordenar por PageRank
            neighbor_scores = [
                (neighbor, self.pagerank.pagerank_scores.get(neighbor, 0))
                for neighbor in neighbors
            ]
            
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            
            return neighbor_scores[:n]
        
        elif method == 'random_walk':
            # Recomendaciones basadas en navegación
            # Simular walks desde el documento
            visits = Counter()
            
            for _ in range(100):
                path = self.random_walk.walk(doc_id, 10)
                visits.update(path)
            
            # Remover el documento actual
            visits.pop(doc_id, None)
            
            return visits.most_common(n)
    
    def get_system_health(self):
        """
        Obtiene métricas de salud del sistema.
        
        Retorna:
        --------
        dict
            Métricas de salud
        """
        # Throughput promedio
        throughputs = [
            doc.get('systemEfficiency', {}).get('throughputRate', 0)
            for doc in self.documents
        ]
        
        # Utilización promedio
        utilizations = [
            doc.get('systemEfficiency', {}).get('systemUtilization', 0)
            for doc in self.documents
        ]
        
        # Queue lengths
        queues = [
            doc.get('systemEfficiency', {}).get('queueLength', 0)
            for doc in self.documents
        ]
        
        return {
            'avg_throughput': np.mean(throughputs),
            'avg_utilization': np.mean(utilizations),
            'avg_queue_length': np.mean(queues),
            'health_score': min(np.mean(utilizations), 1.0)
        }


# Ejemplo de uso
if __name__ == "__main__":
    print("="*60)
    print("DOCUMENT WORKFLOW ANALYZER - DEMOSTRACIÓN")
    print("="*60)
    
    # Cargar datos
    print("\nCargando datos de prueba...")
    
    with DATA_FILE.open("r", encoding="utf-8") as f:
        documents = json.load(f)
    
    print(f"✓ Cargados {len(documents)} documentos")
    
    # Crear analizador
    analyzer = DocumentWorkflowAnalyzer()
    analyzer.load_documents(documents)
    
    # Construir modelos
    analyzer.build_state_transition_model()
    analyzer.build_document_network()
    
    # Análisis 1: Workflow
    workflow_analysis = analyzer.analyze_workflow()
    
    # Análisis 2: Importancia
    importance_analysis = analyzer.analyze_document_importance()
    
    # Análisis 3: Navegación
    navigation_analysis = analyzer.analyze_navigation_patterns(num_simulations=50)
    
    # Salud del sistema
    print("\n" + "="*60)
    print("SALUD DEL SISTEMA")
    print("="*60)
    
    health = analyzer.get_system_health()
    
    print(f"\nMétricas:")
    print(f"  Throughput promedio: {health['avg_throughput']:.1f} docs/hora")
    print(f"  Utilización promedio: {health['avg_utilization']:.1%}")
    print(f"  Longitud de cola promedio: {health['avg_queue_length']:.1f} documentos")
    print(f"  Score de salud: {health['health_score']:.1%}")
    
    # Recomendaciones para un documento
    print("\n" + "="*60)
    print("EJEMPLO DE RECOMENDACIONES")
    print("="*60)
    
    sample_doc = documents[0]['_id']
    print(f"\nRecomendaciones para documento: {sample_doc[:20]}...")
    
    recs = analyzer.generate_recommendations(sample_doc, method='pagerank', n=5)
    
    print("\nTop 5 documentos relacionados:")
    for i, (doc_id, score) in enumerate(recs, 1):
        print(f"  {i}. {doc_id[:20]}... (score: {score:.6f})")
    
    print("\n" + "="*60)
    print("✓ ANÁLISIS COMPLETADO")
    print("="*60)
