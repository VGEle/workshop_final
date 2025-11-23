"""
Generador de Datos Mock para el Sistema de Gestión de Documentos
Genera archivos JSON progresivos para cada unidad del proyecto
"""

import json
import random
from datetime import datetime, timedelta
from faker import Faker

# Inicializar Faker para generar datos realistas
fake = Faker()
Faker.seed(42)  # Semilla para reproducibilidad
random.seed(42)

# Configuración
NUM_DOCUMENTS = 2000  # Número de documentos a generar


class DocumentGenerator:
    """
    Clase para generar documentos con campos progresivos para cada unidad
    """

    def __init__(self, num_docs=2000):
        """
        Inicializa el generador de documentos.

        Parámetros:
        -----------
        num_docs : int
            Número de documentos a generar
        """
        self.num_docs = num_docs
        self.documents = []

    def generate_unit1_base(self):
        """
        Genera la estructura base de documentos para Unidad 1.

        Estructura:
        - _id: Identificador único
        - title: Título del documento
        - content: Contenido (párrafo)
        - tags: Lista de 3 etiquetas
        - lastAccessed: Fecha de último acceso

        Retorna:
        --------
        list : Lista de documentos con estructura Unidad 1
        """
        print("Generando estructura base (Unidad 1)...")

        documents = []

        for i in range(self.num_docs):
            # Generar ID único en formato hexadecimal (similar a MongoDB)
            doc_id = fake.hexify(text="^^^^^^^^^^^^^^^^^^^^^^^^", upper=False)

            # Generar título (una oración)
            title = fake.sentence(nb_words=6).rstrip(".")

            # Generar contenido (un párrafo)
            content = fake.paragraph(nb_sentences=5)

            # Generar 3 tags aleatorios
            tags = [fake.word() for _ in range(3)]

            # Generar fecha de último acceso (últimos 30 días)
            days_ago = random.randint(0, 30)
            last_accessed = (
                datetime.now() - timedelta(days=days_ago)
            ).strftime("%Y-%m-%dT%H:%M:%S Z")

            # Crear documento
            document = {
                "_id": doc_id,
                "title": title,
                "content": content,
                "tags": tags,
                "lastAccessed": last_accessed,
            }

            documents.append(document)

            # Mostrar progreso cada 500 documentos
            if (i + 1) % 500 == 0:
                print(f"  Generados {i + 1}/{self.num_docs} documentos...")

        print(f"✓ Generados {self.num_docs} documentos base")
        return documents

    def add_unit2_fields(self, documents):
        """
        Añade campos de Unidad 2 a los documentos existentes.

        Campos añadidos:
        - documentType: Tipo de documento
        - searchFrequency: Frecuencia de búsqueda

        Parámetros:
        -----------
        documents : list
            Lista de documentos con estructura Unidad 1

        Retorna:
        --------
        list : Lista de documentos con campos de Unidad 2 añadidos
        """
        print("\nAñadiendo campos de Unidad 2 (Data Streams)...")

        document_types = ["report", "memo", "presentation", "email"]

        for i, doc in enumerate(documents):
            # Añadir tipo de documento
            doc["documentType"] = random.choice(document_types)

            # Añadir frecuencia de búsqueda (1-100)
            doc["searchFrequency"] = random.randint(1, 100)

            # Mostrar progreso
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(documents)} documentos...")

        print(f"✓ Añadidos campos de Unidad 2 a {len(documents)} documentos")
        return documents

    def add_unit3_fields(self, documents):
        """
        Añade campos de Unidad 3 (Cadenas de Markov) a los documentos.

        Campos añadidos:
        - documentState: Estado actual del documento
        - previousDocumentState: Estado anterior
        - stateTransitionProb: Probabilidad de transición
        - timeInCurrentState: Tiempo en estado actual (minutos)
        - expectedProcessingTime: Tiempo esperado de procesamiento
        - processingWorkflow: Información del flujo de trabajo
        - classificationAnalysis: Análisis de clasificación
        - retrievalOptimization: Optimización de recuperación
        - lifecycleAnalysis: Análisis del ciclo de vida
        - systemEfficiency: Eficiencia del sistema
        - networkProperties: Propiedades de red
        - stationaryProbability: Probabilidad estacionaria
        - workflowOptimization: Factor de optimización

        Parámetros:
        -----------
        documents : list
            Lista de documentos con estructura Unidad 1 y 2

        Retorna:
        --------
        list : Lista de documentos con campos de Unidad 3 añadidos
        """
        print("\nAñadiendo campos de Unidad 3 (Cadenas de Markov)...")

        states = ["received", "classified", "processed", "archived", "retrieved"]
        priorities = ["low", "normal", "high", "urgent"]
        categories = ["contract", "report", "memo", "presentation", "email", "invoice"]

        for i, doc in enumerate(documents):
            # Estados del documento
            doc["documentState"] = random.choice(states)
            doc["previousDocumentState"] = random.choice(states[:-1])  # No incluir 'retrieved'
            doc["stateTransitionProb"] = round(random.uniform(0.0, 1.0), 3)
            doc["timeInCurrentState"] = random.randint(5, 1440)
            doc["expectedProcessingTime"] = random.randint(10, 480)

            # Flujo de trabajo (processingWorkflow)
            doc["processingWorkflow"] = {
                "workflowStage": random.randint(1, 7),
                "priority": random.choice(priorities),
                "complexityScore": round(random.uniform(0.1, 5.0), 1),
                "processingLoad": round(random.uniform(0.0, 1.0), 2),
                "bottleneckRisk": round(random.uniform(0.0, 1.0), 3),
            }

            # Análisis de clasificación (classificationAnalysis)
            doc["classificationAnalysis"] = {
                "documentCategory": random.choice(categories),
                "classificationConfidence": round(random.uniform(0.0, 1.0), 2),
                "reclassificationProb": round(random.uniform(0.0, 0.3), 3),
                "categoryTransitionMatrix": [
                    round(random.uniform(0.0, 1.0), 3) for _ in range(6)
                ],
            }

            # Optimización de recuperación (retrievalOptimization)
            num_nodes = random.randint(3, 7)
            search_path = []
            for _ in range(num_nodes):
                search_path.append(
                    {
                        "nodeId": random.randint(1, 50),
                        "transitionProbability": round(random.uniform(0.0, 1.0), 3),
                        "pathWeight": round(random.uniform(0.1, 2.0), 2),
                    }
                )

            doc["retrievalOptimization"] = {
                "accessFrequency": random.randint(0, 100),
                "searchPath": search_path,
                "retrievalEfficiency": round(random.uniform(0.0, 1.0), 2),
                "cacheHitProbability": round(random.uniform(0.0, 1.0), 3),
            }

            # Análisis del ciclo de vida (lifecycleAnalysis)
            doc["lifecycleAnalysis"] = {
                "documentAge": random.randint(1, 2160),
                "expectedLifetime": random.randint(168, 8760),
                "retentionProbability": round(random.uniform(0.0, 1.0), 3),
                "archivalTransitionProb": round(random.uniform(0.0, 1.0), 3),
                "obsolescenceRisk": round(random.uniform(0.0, 1.0), 3),
            }

            # Eficiencia del sistema (systemEfficiency)
            doc["systemEfficiency"] = {
                "throughputRate": round(random.uniform(1.0, 100.0), 1),
                "queueLength": random.randint(0, 200),
                "processingBottleneck": random.choice([True, False]),
                "loadBalancingScore": round(random.uniform(0.0, 1.0), 2),
                "systemUtilization": round(random.uniform(0.0, 1.0), 2),
            }

            # Propiedades de red (networkProperties)
            doc["networkProperties"] = {
                "documentNetwork": {
                    "nodeId": random.randint(1, 100),
                    "networkDegree": random.randint(2, 15),
                    "clusteringCoefficient": round(random.uniform(0.0, 1.0), 3),
                    "centralityScore": round(random.uniform(0.0, 1.0), 3),
                },
                "randomWalkProperties": {
                    "coverTime": random.randint(10, 500),
                    "mixingTime": random.randint(5, 100),
                    "hittingTime": random.randint(3, 50),
                },
            }

            # Propiedades adicionales
            doc["stationaryProbability"] = round(random.uniform(0.0, 1.0), 3)
            doc["workflowOptimization"] = round(random.uniform(0.7, 1.3), 2)

            # Mostrar progreso
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(documents)} documentos...")

        print(f"✓ Añadidos campos de Unidad 3 a {len(documents)} documentos")
        return documents

    def add_unit4_fields(self, documents):
        """
        Añade campos de Unidad 4 (MapReduce) a los documentos.

        Campos añadidos:
        - mapReducePartition: Número de partición
        - processingNode: Nodo de procesamiento
        - batchId: ID del lote
        - aggregationKey: Clave de agregación

        Parámetros:
        -----------
        documents : list
            Lista de documentos con estructura Unidad 1, 2 y 3

        Retorna:
        --------
        list : Lista de documentos con campos de Unidad 4 añadidos
        """
        print("\nAñadiendo campos de Unidad 4 (MapReduce)...")

        for i, doc in enumerate(documents):
            # Campos de MapReduce
            doc["mapReducePartition"] = random.randint(1, 25)
            doc["processingNode"] = f"node_{random.randint(1, 12)}"
            doc["batchId"] = f"batch_{random.randint(1000, 9999)}"

            # Clave de agregación basada en documentType
            # Necesitamos un departamento ficticio
            departments = ["HR", "Finance", "Engineering", "Marketing", "Operations"]
            department = random.choice(departments)
            doc["aggregationKey"] = f"{doc['documentType']}_{department}"

            # Mostrar progreso
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(documents)} documentos...")

        print(f"✓ Añadidos campos de Unidad 4 a {len(documents)} documentos")
        return documents

    def add_unit5_fields(self, documents):
        """
        Añade campos de Unidad 5 (Near Neighbor Search) a los documentos.

        Campos añadidos:
        - documentSimilarity: Información de similitud del documento
          - accessMetrics: Array de 4 métricas de acceso
          - documentType: Tipo de documento
          - department: Departamento
          - documentSize: Tamaño del documento (KB)
          - relevanceScore: Puntuación de relevancia
          - userAccessPattern: Patrón de acceso de usuarios

        Parámetros:
        -----------
        documents : list
            Lista de documentos con estructura Unidad 1, 2, 3 y 4

        Retorna:
        --------
        list : Lista de documentos con campos de Unidad 5 añadidos
        """
        print("\nAñadiendo campos de Unidad 5 (Near Neighbor Search)...")

        doc_types_similarity = ["report", "manual", "policy", "training"]
        departments = ["HR", "Finance", "Engineering", "Marketing"]

        for i, doc in enumerate(documents):
            # Métricas de acceso (4 valores)
            access_metrics = [random.randint(1, 50) for _ in range(4)]

            # Información de similitud
            doc["documentSimilarity"] = {
                "accessMetrics": access_metrics,
                "documentType": random.choice(doc_types_similarity),
                "department": random.choice(departments),
                "documentSize": random.randint(10, 500),  # KB
                "relevanceScore": round(random.uniform(0.3, 1.0), 2),
                "userAccessPattern": round(random.uniform(0.1, 1.0), 2),
            }

            # Mostrar progreso
            if (i + 1) % 500 == 0:
                print(f"  Procesados {i + 1}/{len(documents)} documentos...")

        print(f"✓ Añadidos campos de Unidad 5 a {len(documents)} documentos")
        return documents

    def save_to_json(self, documents, filename):
        """
        Guarda la lista de documentos en un archivo JSON.

        Parámetros:
        -----------
        documents : list
            Lista de documentos a guardar
        filename : str
            Nombre del archivo de salida
        """
        filepath = f"data/{filename}"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)

        print(f"✓ Archivo guardado: {filepath}")

        # Calcular y mostrar tamaño del archivo
        import os

        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        print(f"  Tamaño: {size_mb:.2f} MB")

    def generate_all_units(self):
        """
        Genera todos los archivos JSON para todas las unidades.

        Este método ejecuta todo el proceso de generación:
        1. Genera estructura base (Unidad 1)
        2. Añade campos progresivamente para cada unidad
        3. Guarda un archivo JSON separado para cada unidad
        """
        print("=" * 70)
        print("GENERADOR DE DATOS MOCK - SISTEMA DE GESTIÓN DE DOCUMENTOS")
        print("=" * 70)

        # Generar base (Unidad 1)
        documents_unit1 = self.generate_unit1_base()
        self.save_to_json(documents_unit1, "unit1_documents.json")

        # Unidad 2 (copia de Unidad 1 + nuevos campos)
        documents_unit2 = [doc.copy() for doc in documents_unit1]
        documents_unit2 = self.add_unit2_fields(documents_unit2)
        self.save_to_json(documents_unit2, "unit2_documents.json")

        # Unidad 3 (copia de Unidad 2 + nuevos campos)
        documents_unit3 = [doc.copy() for doc in documents_unit2]
        documents_unit3 = self.add_unit3_fields(documents_unit3)
        self.save_to_json(documents_unit3, "unit3_documents.json")

        # Unidad 4 (copia de Unidad 3 + nuevos campos)
        documents_unit4 = []
        for doc in documents_unit3:
            # Hacer copia profunda para evitar modificar el original
            import copy

            documents_unit4.append(copy.deepcopy(doc))
        documents_unit4 = self.add_unit4_fields(documents_unit4)
        self.save_to_json(documents_unit4, "unit4_documents.json")

        # Unidad 5 (copia de Unidad 4 + nuevos campos)
        documents_unit5 = []
        for doc in documents_unit4:
            import copy

            documents_unit5.append(copy.deepcopy(doc))
        documents_unit5 = self.add_unit5_fields(documents_unit5)
        self.save_to_json(documents_unit5, "unit5_documents.json")

        print("\n" + "=" * 70)
        print("✓ GENERACIÓN COMPLETADA")
        print("=" * 70)
        print(f"Total de archivos generados: 5")
        print(f"Documentos por archivo: {self.num_docs}")
        print("\nArchivos creados:")
        print("  - data/unit1_documents.json")
        print("  - data/unit2_documents.json")
        print("  - data/unit3_documents.json")
        print("  - data/unit4_documents.json")
        print("  - data/unit5_documents.json")


# Punto de entrada principal
if __name__ == "__main__":
    # Crear instancia del generador
    generator = DocumentGenerator(num_docs=NUM_DOCUMENTS)

    # Generar todos los archivos
    generator.generate_all_units()
