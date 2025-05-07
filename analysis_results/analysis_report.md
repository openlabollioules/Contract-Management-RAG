# RAG Performance Analysis Report

## Test Configuration

- **Date:** 2025-04-30 10:36:40
- **Model:** mistral-small3.1:latest
- **Description:** basic test

## Performance Summary

### Overall Metrics

| Metric | Basic RAG | GraphRAG | Difference |
|--------|-----------|----------|------------|
| Average Response Time | 26.56s | 28.47s | 1.91s (7.2%) |
| Total Questions | 25 | 25 | - |

**Interpretation:** GraphRAG shows a significant increase in response time compared to Basic RAG.

### Source Usage Analysis

- **Average sources per question:** Basic RAG: 7.0, GraphRAG: 10.0 (3.0 difference, 42.3%)

**Source Type Distribution:**

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 14.9% |
| document | 100.0% | 85.1% |

### Performance by Question Category

| Category | Basic RAG (s) | GraphRAG (s) | Difference (s) | Difference (%) |
|----------|---------------|--------------|----------------|----------------|
| factual | 26.24 | 43.76 | 17.52 | 66.8% |
| relational | 28.49 | 27.39 | -1.10 | -3.9% |
| complex | 24.82 | 24.95 | 0.12 | 0.5% |
| cross_reference | 27.52 | 24.41 | -3.12 | -11.3% |
| definitions | 25.73 | 21.83 | -3.90 | -15.2% |

**Best performing categories with GraphRAG:**

- **definitions**: 15.2% faster than Basic RAG
- **cross_reference**: 11.3% faster than Basic RAG
- **relational**: 3.9% faster than Basic RAG

**Worst performing categories with GraphRAG:**

- **factual**: 66.8% slower than Basic RAG
- **complex**: 0.5% slower than Basic RAG
- **relational**: 3.9% faster than Basic RAG

## Visualizations

The following visualizations have been generated:

1. **Response Time Comparison**: Bar chart comparing average response times by question category.
2. **Sources Count Comparison**: Bar chart comparing the number of sources used by each RAG method.
3. **Source Types Comparison**: Stacked bar chart showing the distribution of source types used.
4. **Category Performance Heatmap**: Heatmap showing detailed performance across question categories.

Images are available in the same directory as this report.

## Conclusion

Based on the analysis, GraphRAG shows slower performance compared to Basic RAG with an average response time increase of 7.2%. GraphRAG performs particularly well on definitions questions but struggles more with factual questions compared to Basic RAG.

GraphRAG successfully leverages graph-based sources (14.9% of all sources) to enhance retrieval, which might contribute to its performance characteristics.

## Detailed Source Analysis

### Most Frequently Used Documents

| Document | Frequency |
|----------|----------|
|  | 237 |
| ### 1. APPLICABILITY AND DEFINITIONS (CGA_E_oriented.pdf) | 66 |
| Referentiel_B_oriented (Referentiel_B_oriented.pdf) | 53 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 16 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 9 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 9 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 7 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 6 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 5 |
| - 1.2 **"Incoming Inspection"** is an inspection of qualitat... | 4 |

### Source Type Distribution by Question Category

#### Factual Questions

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 15.7% |
| document | 100.0% | 84.3% |

#### Relational Questions

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 15.7% |
| document | 100.0% | 84.3% |

#### Complex Questions

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 17.0% |
| document | 100.0% | 83.0% |

#### Cross_Reference Questions

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 14.3% |
| document | 100.0% | 85.7% |

#### Definitions Questions

| Source Type | Basic RAG | GraphRAG |
|-------------|-----------|----------|
| graph | 0.0% | 11.1% |
| document | 100.0% | 88.9% |


### Semantic Distance Analysis

| Method | Min Distance | Max Distance | Mean Distance | Median Distance |
|--------|--------------|--------------|---------------|----------------|
| Basic RAG | 0.2788 | 0.5141 | 0.4058 | 0.4064 |
| GraphRAG | 0.0000 | 0.5141 | 0.2878 | 0.3733 |

### Source Reuse Analysis

- **Basic RAG**:
  - Average questions per document: 5.77
  - Maximum questions using same document: 25
  - Percentage of documents used in multiple questions: 53.8%

- **GraphRAG**:
  - Average questions per document: 5.38
  - Maximum questions using same document: 25
  - Percentage of documents used in multiple questions: 56.2%


### Additional Source Visualizations

The following additional visualizations have been generated:

1. **Top Documents**: Bar chart showing the most frequently used documents.
2. **Distance Distribution**: Histogram showing the distribution of semantic distances.
3. **Source Types by Category**: Charts showing source type usage patterns for each question category.

Images are available in the same directory as this report.

## Conclusion

Based on the analysis, GraphRAG shows slower performance compared to Basic RAG with an average response time increase of 7.2%. GraphRAG performs particularly well on definitions questions but struggles more with factual questions compared to Basic RAG.

GraphRAG successfully leverages graph-based sources (14.9% of all sources) to enhance retrieval, which might contribute to its performance characteristics.
