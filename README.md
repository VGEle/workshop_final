## Automated Document Sorting and Retrieval System

**Proyecto Final - Lógica y Representación III**

## Equipo
- Carolina Gómez Osorno
- Elena Vargas Grisales
- Jonathan Arley Alzate Castaño

**Profesor:** William Cornejo  
**Fecha:** Noviembre 2025

---

## Descripción

Sistema completo de gestión de documentos que implementa **18 algoritmos** de Big Data distribuidos en **5 unidades**:

| Unidad | Tema | Algoritmos | Documentos |
|--------|------|-----------|------------|
| 1 | Hashing | 3 | 2,000 |
| 2 | Data Streams | 5 | 2,000 |
| 3 | Cadenas de Markov | 3 | 2,000 |
| 4 | MapReduce | 3 | 2,000 |
| 5 | Near Neighbor Search | 4 | 2,000 |

**Total:** 18 algoritmos procesando 10,000 documentos

---

## Inicio Rápido (5 minutos)

### 0. Ruta de trabajo
Sitúate dentro de la carpeta del proyecto: `proyecto_gestion_documentos`.  
Todos los comandos asumen que estás allí (evita errores de rutas, sobre todo en Windows).

### 1. Requisitos Previos

- Python 3.9+
- Entorno virtual

### 2. Instalación (macOS/Linux/Windows)

```bash
# Clonar o descomprimir y entrar
cd proyecto_gestion_documentos

# Crear entorno virtual (en la carpeta del proyecto)
python -m venv venv

# Activar venv
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell o CMD)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Generar Datos Mock (opcional si ya tienes los JSON)

```bash
cd data
python generate_mock_data.py
cd ..
```

### 4. Ejecutar Unidades (desde la raíz del proyecto)

```bash
# Unit 1: Hashing & Search
python unit1_hashing/load_and_test.py

# Unit 2: Data Streams
python unit2_datastreams/bloom_filter.py

# Unit 3: Markov Chains
python unit3_markov/document_workflow_analyzer.py

# Unit 4: MapReduce
python unit4_mapreduce/document_analyzer.py

# Unit 5: Similarity Search
python unit5_similarity/similarity_analyzer.py
```

> Nota: Los scripts ahora construyen la ruta a los JSON con `Path(__file__).resolve()`, así que funcionarán incluso si ejecutas desde otra carpeta, pero mantenerte en `proyecto_gestion_documentos` evita confusiones.

---

## Estructura del Proyecto

```
proyecto_gestion_documentos/
├── data/                          # 10,000 documentos JSON
│   ├── unit1_documents.json
│   ├── unit2_documents.json
│   ├── unit3_documents.json
│   ├── unit4_documents.json
│   └── unit5_documents.json
│
├── documentation/                 # Documentación
│   ├── unit1_theory.md
│   └── unidades_1_y_2_resumen.md
│
├── unit1_hashing/                # Hashing & Search
│   ├── hash_table.py
│   ├── document_management_system.py
│   ├── search_engine.py
│   ├── recommendation_system.py
│   └── load_and_test.py
│
├── unit2_datastreams/            # 5 Algoritmos de Streams
│   ├── bloom_filter.py
│   ├── reservoir_sampling.py
│   ├── distinct_counting.py
│   ├── frequency_moments.py
│   └── dgim_algorithm.py
│
├── unit3_markov/                 # Markov Chains
│   ├── markov_chain.py
│   ├── pagerank.py
│   ├── random_walk.py
│   └── document_workflow_analyzer.py
│
├── unit4_mapreduce/              # MapReduce
│   ├── mapreduce_framework.py
│   ├── document_analyzer.py
│   └── cost_analyzer.py
│
├── unit5_similarity/             # Near Neighbor Search
│   ├── jaccard_similarity.py
│   ├── minhash.py
│   ├── lsh.py
│   └── similarity_analyzer.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Unidades Detalladas

### Unit 1: Hashing (Hash Tables, Search, Recommendations)

Implementa estructuras de hashing para acceso rápido a documentos.

**Archivos:**
- `hash_table.py` - Hash table con manejo de colisiones
- `document_management_system.py` - Sistema completo de gestión
- `search_engine.py` - Motor de búsqueda rápida
- `recommendation_system.py` - Recomendaciones basadas en hashing

**Características:**
- Hash table con ~99.2% de precisión
- Búsqueda O(1) en promedio
- Recomendaciones por similitud de hash

**Ejecutar:**
```bash
python unit1_hashing/load_and_test.py
```

---

### Unit 2: Data Streams (5 Algoritmos)

Procesa flujos continuos de datos sin almacenar todo en memoria.

**Algoritmos:**

1. **Bloom Filter**
   - Membresía de conjuntos con falsos positivos
   - 99.2% de precisión, 0.8% FP rate
   ```bash
   python unit2_datastreams/bloom_filter.py
   ```

2. **Reservoir Sampling**
   - Muestreo uniforme de flujos infinitos
   - 99.29% de uniformidad
   ```bash
   python unit2_datastreams/reservoir_sampling.py
   ```

3. **Distinct Counting**
   - Cuenta elementos únicos exactamente
   - Análisis de documentos por usuario
   ```bash
   python unit2_datastreams/distinct_counting.py
   ```

4. **Frequency Moments**
   - Estadísticas de distribución (F0, F1, F2)
   - Cálculo de entropía y coeficiente Gini
   ```bash
   python unit2_datastreams/frequency_moments.py
   ```

5. **DGIM Algorithm**
   - Conteo de bits en ventana deslizante
   - Compresión 45,000x del espacio
   ```bash
   python unit2_datastreams/dgim_algorithm.py
   ```

---

### Unit 3: Cadenas de Markov

Modela transiciones entre estados de documentos.

**Componentes:**
- `markov_chain.py` - Cadena de Markov
- `pagerank.py` - Algoritmo PageRank
- `random_walk.py` - Paseo aleatorio
- `document_workflow_analyzer.py` - Análisis completo

**Características:**
- Transiciones entre estados de documentos
- Importancia relativa (PageRank)
- Simulación de navegación

**Ejecutar:**
```bash
python unit3_markov/document_workflow_analyzer.py
```

---

### Unit 4: MapReduce

Framework de procesamiento distribuido (implementación secuencial).

**Componentes:**
- `mapreduce_framework.py` - Framework base
- `document_analyzer.py` - 5 análisis de documentos
- `cost_analyzer.py` - Análisis de costos

**Análisis Disponibles:**
1. Word Count - Palabras más frecuentes
2. Document Statistics - Estadísticas por categoría
3. Tag Analysis - Distribución de tags
4. State Transitions - Transiciones entre estados
5. Efficiency Analysis - Métricas del sistema

**Ejecutar:**
```bash
python unit4_mapreduce/document_analyzer.py
python unit4_mapreduce/cost_analyzer.py
```

**Características:**
- Ejecución secuencial sin multiprocessing
- Seguimiento de tiempo por fase
- Estimación de costos (comunicación, CPU, almacenamiento)

---

### Unit 5: Near Neighbor Search

Detección eficiente de documentos similares.

**Algoritmos:**

1. **Jaccard Similarity & Shingling**
   - Similitud de conjuntos: J(A, B) = |A ∩ B| / |A ∪ B|
   - k-shingles de caracteres o palabras
   ```bash
   python unit5_similarity/jaccard_similarity.py
   ```

2. **MinHash**
   - Firmas compactas que preservan similitud
   - Compresión 100x con error <5%
   ```bash
   python unit5_similarity/minhash.py
   ```

3. **Locality Sensitive Hashing (LSH)**
   - Búsqueda sin comparar todos los pares
   - Aceleración: O(n) vs O(n²)
   - Reducción: 99%+ pares evitados
   ```bash
   python unit5_similarity/lsh.py
   ```

4. **Similarity Analyzer** (Sistema Completo)
   - Detección de duplicados (threshold: 0.9)
   - Búsqueda por categoría (threshold: 0.7)
   - Análisis de distribución
   - Reportes de eficiencia
   ```bash
   python unit5_similarity/similarity_analyzer.py
   ```

---

## Dependencias

```
numpy==1.24.3           # Computaciones numéricas
pandas==2.0.3           # Manipulación de datos
faker==19.3.1           # Generación de datos mock
matplotlib==3.7.2       # Visualización (preparado)
seaborn==0.12.2         # Visualización estadística
mmh3==4.0.1             # Funciones hash
pytest==7.4.0           # Testing
pytest-cov==4.1.0       # Coverage
```

Instalar con:
```bash
pip install -r requirements.txt
```

---

## Resultados y Métricas

### Precisión

| Algoritmo | Métrica | Resultado |
|-----------|---------|-----------|
| Bloom Filter | Precisión | 99.2% |
| Reservoir Sampling | Uniformidad | 99.29% |
| Distinct Counting | Exactitud | 100% |
| Frequency Moments | Correlación | >0.99 |
| DGIM | Compresión | 45,000x |
| Jaccard | Similitud | Exacta |
| MinHash | Error | <5% |
| LSH | Aceleración | >100x |

### Performance

| Operación | Documentos | Tiempo |
|-----------|-----------|--------|
| Indexar (Unit 1) | 2,000 | <0.5s |
| Bloom Filter | 500 | <0.1s |
| Reservoir Sampling | 1,000 | <0.2s |
| MapReduce Jobs | 2,000 | ~0.5s |
| Índice LSH | 2,000 | <1.0s |
| Búsqueda LSH | 2,000 | <100ms |

---

## Uso Avanzado

### Personalizar Parámetros

Unit 2 - Bloom Filter:
```python
from unit2_datastreams.bloom_filter import BloomFilter

bf = BloomFilter(size=10000, num_hashes=5)
bf.add('elemento')
```

Unit 5 - LSH:
```python
from unit5_similarity.lsh import LSHDocumentIndex

index = LSHDocumentIndex(num_hashes=200, num_bands=10)
# Menor bandas = mayor threshold pero más velocidad
```

### Generar Reportes

Cada módulo genera reportes con:
- Estadísticas de entrada/salida
- Métricas de rendimiento
- Análisis de eficiencia
- Recomendaciones de optimización

---

## Testing

```bash
# Ejecutar tests de Unit 1
pytest unit1_hashing/test_basic.py -v

# Ejecutar todos los tests
pytest --cov=. --cov-report=html
```

---

## Notas Importantes

1. **Datos Mock**: Los archivos JSON se generan automáticamente con `generate_mock_data.py`
2. **Configuración Git**: `.gitignore` excluye automáticamente venv, __pycache__, y archivos temporales
3. **Virtual Environment**: Ubicado en nivel superior `../venv` (separado del proyecto)
4. **Secuencial**: Unit 4 (MapReduce) usa ejecución secuencial para compatibilidad
5. **Compresión**: Unit 5 logra 45,000x+ compresión sin perder similitud

---

## Licencia

Proyecto educativo - Lógica y Representación III  
Universidad de Antioquia - Facultad de Ingeniería

---

**Última actualización:** 23 de noviembre de 2025
