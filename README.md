# Entity Resolution for Yale University Library Catalog

## Overview

This project implements a scalable entity resolution pipeline for the Yale University Library catalog, designed to disambiguate and cluster personal name entities across catalog records. The system leverages vector embeddings, approximate nearest neighbor search, and machine learning classification to resolve entities at scale.

## Key Features

- **Vector-Based Entity Resolution**: Uses 1,536-dimensional embeddings from OpenAI's text-embedding-3-small model
- **Efficient Deduplication**: Minimizes redundancy by deduplicating strings before embedding
- **Vector Database Integration**: Leverages Weaviate for efficient similarity search and persistence
- **Intelligent Null Value Imputation**: Uses vector-based "hot deck" imputation for missing fields
- **Feature-Rich Classification**: Employs sophisticated feature engineering and logistic regression classification
- **Graph-Based Clustering**: Applies community detection and transitivity for entity grouping
- **Scalable Architecture**: Scales from development (8 cores, 32GB RAM) to production (64 cores, 256GB RAM)
- **Modular Design**: Separate components for preprocessing, embedding, indexing, etc.

## System Requirements

- Python 3.9+
- Docker and Docker Compose
- 32GB+ RAM (Development), 256GB+ RAM (Production)
- 8+ CPU cores (Development), 64+ CPU cores (Production)
- 50GB+ disk space

## Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/yale-library/entity-resolution.git
cd entity-resolution
```

2. **Set up environment**

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Start Weaviate**

```bash
docker-compose up -d
```

4. **Configure the pipeline**

Edit `config.yml` to adjust parameters as needed.

5. **Run the pipeline**

```bash
# Development mode (processes a subset of data)
python main.py --mode dev

# Production mode (processes all data)
python main.py --mode prod
```

## Pipeline Stages

The entity resolution pipeline consists of the following stages:

1. **Data Preprocessing**: Normalizes and deduplicates text fields, tracks frequency of duplicate strings
2. **Vector Embedding**: Generates embeddings for deduplicated strings using OpenAI's text-embedding-3-small model
3. **Weaviate Indexing**: Indexes embeddings in Weaviate for efficient similarity search
4. **Null Value Imputation**: Applies vector-based "hot deck" approach to impute null values
5. **Feature Engineering**: Constructs feature vectors for record pairs, including similarity metrics and interaction features
6. **Classifier Training**: Trains a logistic regression classifier using gradient descent
7. **Entity Clustering**: Uses graph-based community detection to group matches into entity clusters
8. **Evaluation and Analysis**: Evaluates precision, recall, and analyzes error patterns

## Configuration

The pipeline is configured via `config.yml`. Key configuration parameters include:

- **Resource allocation**: Controls batch sizes, parallelism, and memory usage
- **Model parameters**: Embedding model, classifier settings, etc.
- **Feature engineering**: Similarity metrics, interaction features, etc.
- **Weaviate settings**: Connection parameters, index settings, etc.

## Directory Structure

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
   │   └── utils.py            # Utility functions
   ├── notebooks/              # Analysis notebooks
   │   ├── evaluation.ipynb    # Results evaluation
   │   └── exploration.ipynb   # Results exploration
   └── tests/                  # Testing scripts
       └── test_pipeline.py    # Pipeline verification
```

## Execution Workflow

The complete pipeline execution workflow is:

1. **Preprocessing**: Read CSV files, extract and deduplicate fields, maintain hash-based data structures
2. **Embedding**: Generate vectors for unique strings, store them in Weaviate
3. **Indexing**: Create efficient indexes in Weaviate for similarity search
4. **Imputation**: Impute missing values using vector-based hot deck approach
5. **Training**: Train classifier on labeled data
6. **Classification**: Apply classifier to full dataset
7. **Clustering**: Group matches into entity clusters
8. **Evaluation**: Analyze results, generate reports

Each stage can be run independently or as part of the complete pipeline.

## Ground Truth Data

The system uses a labeled dataset for training and testing:

- **Training/testing dataset**: 2,000+ labeled records with ground truth matches
- **Complete dataset**: 600+ CSV files for full classification (~15GB data)

## Monitoring

The pipeline includes Prometheus integration for monitoring resource usage, processing time, and other metrics.

## License

[License information]

## Contributors

[Contributor information]
