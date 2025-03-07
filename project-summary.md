# Entity Resolution for Yale University Library Catalog - Project Summary

## Overview

This document provides a comprehensive summary of the entity resolution system developed for the Yale University Library catalog. The system is designed to disambiguate and cluster personal name entities across catalog records using vector embeddings, approximate nearest neighbor search, and machine learning classification.

## System Architecture

The entity resolution pipeline consists of the following components:

1. **Data Preprocessing**: Extracts and deduplicates fields from the dataset, maintaining hash-based data structures for efficient storage and retrieval.

2. **Vector Embedding**: Generates 1,536-dimensional vector embeddings for unique strings using OpenAI's text-embedding-3-small model.

3. **Weaviate Indexing**: Indexes embeddings in Weaviate for efficient similarity search, with named vectors for each field type.

4. **Null Value Imputation**: Uses a vector-based hot deck approach to impute missing values, improving data completeness.

5. **Candidate Retrieval**: Uses the `person` vector as a blocking key to retrieve match candidates via approximate nearest neighbor search.

6. **Feature Engineering**: Constructs feature vectors for record pairs, including similarity metrics and interaction features.

7. **Classification**: Trains a logistic regression classifier using gradient descent to determine whether pairs represent the same entity.

8. **Clustering**: Groups matches into entity clusters using graph-based community detection.

9. **Analysis**: Analyzes pipeline results, identifying patterns, anomalies, and insights.

10. **Reporting**: Generates reports and visualizations from pipeline results.

## Key Technologies

- **Vector Embeddings**: OpenAI's text-embedding-3-small (1,536 dimensions)
- **Vector Database**: Weaviate (v1.24.x) with HNSW algorithm for ANN search
- **Classification**: Logistic regression with gradient descent
- **Clustering**: Connected components and community detection algorithms
- **Parallelization**: Process-based parallelism for batch processing
- **Persistence**: Memory-mapped files for efficient storage and checkpointing

## Implementation Details

### Data Structures

The system uses several key data structures for efficient processing:

- **Hash-based deduplication**: Minimizes redundancy by hashing and deduplicating text fields
- **Record-field hashes**: Maps record IDs to field hashes for efficient lookup
- **Field-hash mapping**: Tracks which fields contain which hash values
- **String counts**: Monitors frequency of duplicate strings
- **Memory-mapped dictionaries**: Enables handling of large datasets with limited memory

### Feature Engineering

The system employs sophisticated feature engineering:

- **Direct similarity features**: Vector cosine similarity for each field
- **String similarity features**: Levenshtein distance, Jaro-Winkler similarity
- **Interaction features**: Harmonic means, products, ratios between field similarities
- **Birth/death year features**: Match status of extracted birth/death years
- **Prefilters**: Automatic classification based on configurable rules

### Optimization Techniques

- **Batched processing**: Data is processed in configurable batches
- **Parallel execution**: Utilizes multiple cores for improved throughput
- **Checkpoint mechanisms**: Enables resuming interrupted processing
- **Scalable architecture**: Adapts to available hardware resources
- **Configurable parameters**: Central configuration for tuning performance

### Monitoring and Analysis

- **Prometheus integration**: Real-time monitoring of resource usage and throughput
- **Comprehensive metrics**: Precision, recall, F1-score, confusion matrix
- **Feature importance analysis**: Identifies most influential features
- **Cluster statistics**: Size distribution, entity relationships
- **Visualization**: Interactive notebooks for exploring results

## Performance Considerations

### Resource Requirements

- **Development**: 8 cores, 32GB RAM (recommended minimum)
- **Production**: 64 cores, 256GB RAM (recommended for full dataset)
- **Storage**: Approximately 50GB for processing 600+ CSV files (~15GB raw data)
- **API Usage**: Respects OpenAI rate limits (5,000,000 tokens/minute, 500,000,000 tokens/day)

### Scalability

The system is designed to scale from development to production:

- **Memory optimization**: Uses memory-mapped files for large datasets
- **Configurable batch size**: Adjusts to available memory
- **Parallel processing**: Distributes workload across available cores
- **Development mode**: Processes subset of data for rapid iteration
- **Checkpointing**: Enables resuming from interruptions

## Special Considerations

### Birth/Death Year Extraction

A specialized module (`birth_death_year_regexes.py`) handles extraction of birth and death years from person names, which is crucial for entity disambiguation:

- Comprehensive pattern matching for various date formats
- Handles approximate dates ("circa", "ca.", etc.)
- Normalizes year values for consistent comparison
- Provides confidence values for extracted years

### Handling Temporal Ambiguity

The system is designed to avoid overly rigid temporal reasoning:

- Recognizes that publication dates do not strictly constrain person lifespans
- Uses temporal information as a signal rather than a hard constraint
- Implements configurable thresholds for temporal overlap calculations

### Multilingual Support

The system is designed to handle multilingual data:

- Vector embeddings operate on semantic meaning, not specific languages
- No language-specific rules or heuristics that would bias against non-English data
- Text normalization preserves important cultural and linguistic information

## Deployment Instructions

1. **Set up environment**:
   ```bash
   ./setup.sh
   ```

2. **Configure parameters** in `config.yml` according to hardware capabilities and requirements.

3. **Run in development mode** for testing:
   ```bash
   python main.py --mode dev
   ```

4. **Run in production mode** for full dataset:
   ```bash
   python main.py --mode prod
   ```

5. **Monitor progress** via Prometheus metrics and logs.

6. **Analyze results** using the provided notebooks and reports.

## Project Structure

```
entity-resolution/
   ├── README.md               # Project documentation
   ├── config.yml              # Configuration parameters
   ├── docker-compose.yml      # Docker Compose for Weaviate
   ├── prometheus.yml          # Prometheus monitoring configuration
   ├── requirements.txt        # Python dependencies
   ├── main.py                 # Entry point script
   ├── setup.sh                # Setup script
   ├── src/                    # Source code
   │   ├── batch_preprocessing.py    # Data preprocessing
   │   ├── embedding.py        # Vector embedding
   │   ├── indexing.py         # Weaviate integration
   │   ├── imputation.py       # Null value imputation
   │   ├── batch_querying.py   # Querying and match candidate retrieval
   │   ├── parallel_features.py # Feature engineering
   │   ├── classification.py   # Classifier training/evaluation
   │   ├── clustering.py       # Entity clustering
   │   ├── pipeline.py         # Pipeline orchestration
   │   ├── analysis.py         # Analysis of pipeline processes and results
   │   ├── reporting.py        # Reporting and visualization of pipeline results
   │   ├── birth_death_year_regexes.py # Birth/death year extraction
   │   └── utils.py            # Utility functions
   ├── notebooks/              # Analysis notebooks
   │   ├── evaluation.ipynb    # Results evaluation
   │   └── exploration.ipynb   # Results exploration
   └── tests/                  # Testing scripts
       └── test_pipeline.py    # Pipeline verification
```

## Future Enhancements

Potential areas for future improvement include:

1. **Additional embedding models**: Support for alternative embedding models beyond OpenAI
2. **Advanced classification algorithms**: Experimentation with neural networks or ensemble methods
3. **Interactive dashboard**: Web-based dashboard for real-time monitoring and result exploration
4. **Incremental processing**: Support for processing new records without full recomputation
5. **Active learning**: User feedback integration for continuous improvement

## Conclusion

This entity resolution system provides a robust, scalable solution for disambiguating and clustering personal name entities in the Yale University Library catalog. By leveraging vector embeddings, approximate nearest neighbor search, and machine learning classification, it achieves high accuracy while maintaining computational efficiency.

The modular architecture and configurable parameters enable adaptation to different hardware environments and dataset characteristics, making it suitable for both development and production use cases.
