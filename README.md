# Enhanced File Clustering for RAG Models

A sophisticated document clustering tool that uses Google Gemini to create hierarchical groupings of files centered around themes. This tool is particularly useful for organizing large document collections and preparing them for RAG (Retrieval-Augmented Generation) systems.

## Features

- **Hierarchical Topic Modeling**: Creates both global and local topic clusters for better organization
- **Smart Document Processing**:
  - Sliding window chunking (1000-1500 tokens per chunk)
  - Configurable chunk overlap
  - Support for multiple file formats (PDF, DOCX, PPTX, MD)
- **Advanced Embedding Techniques**:
  - Google Gemini embeddings
  - Optional contrastive learning for improved embeddings
  - Automatic optimal cluster determination using silhouette scores
- **Intelligent Cluster Naming**: Uses Gemini to generate meaningful cluster names based on content

## Requirements

- Python 3.x
- Key Dependencies:
  ```
  google-generativeai
  llama-index
  scikit-learn
  numpy
  tqdm
  tiktoken
  sentence-transformers
  pydantic
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google Gemini API key

## Usage

### Basic Usage

```bash
python src/enhanced_file_clustering.py /path/to/input/directory /path/to/output.json
```

### Configuration

The tool uses Pydantic for configuration. Key parameters include:

```python
{
    "input_dir": "Directory containing files to cluster",
    "output_file": "Path to output JSON file",
    "api_key": "Google Gemini API key",
    "max_pages": 10,  # Maximum pages per document
    "chunk_size": 1500,  # Size of chunks in tokens
    "chunk_overlap": 300,  # Overlap between chunks
    "use_contrastive_learning": true,
    "min_local_clusters": 2,
    "max_local_clusters": 20,
    "min_global_clusters": 2,
    "max_global_clusters": 15,
    "contrastive_model_name": "all-MiniLM-L6-v2",
    "temperature": 0.07
}
```

## Output Format

The tool generates a JSON file with a hierarchical structure:

```json
{
    "Global Category 1": {
        "Local Category 1": ["file1.pdf", "file2.docx"],
        "Local Category 2": ["file3.pdf"]
    },
    "Global Category 2": {
        "Local Category 3": ["file4.md", "file5.pptx"]
    }
}
```

## Supported File Types

- PDF (`.pdf`)
- Microsoft Word (`.docx`, `.doc`)
- Microsoft PowerPoint (`.pptx`, `.ppt`)
- Markdown (`.md`)

## Advanced Features

### Contrastive Learning

When enabled, the tool uses contrastive learning to improve embedding quality by:
- Creating augmented views of text chunks
- Applying contrastive loss
- Generating more robust and discriminative embeddings

### Optimal Clustering

The tool automatically determines the optimal number of clusters by:
- Using silhouette scores for cluster evaluation
- Testing different cluster numbers within configured ranges
- Adapting to the size and nature of the document collection

## Limitations

- Requires Google Gemini API access
- Large documents are processed with a page limit (default: 10 pages)
- Token limits apply to individual chunks
- Some features require optional dependencies (e.g., sentence-transformers for contrastive learning)

## License

[Add your license information here]
