#!/usr/bin/env python3
"""
Enhanced File Clustering Script for RAG Model

This script uses Google Gemini to create hierarchical groupings of files centered around themes.
It processes documents using sliding window chunking and contrastive learning to improve clustering.

Features:
- Sliding window chunking for large documents (1000-1500 tokens per chunk)
- Contrastive learning for improved embeddings
- Hierarchical topic modeling (local to global topics)
- Pydantic configuration

Requirements:
    - llama-index
    - google-generativeai
    - scikit-learn
    - numpy
    - tqdm
    - tiktoken (for tokenization)
    - sentence-transformers (for contrastive learning)
    - pydantic (for configuration)
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple, Any, Union
import logging
from collections import defaultdict
import re

# For progress tracking
from tqdm import tqdm

# For embeddings and clustering
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Google Gemini API
import google.generativeai as genai

# LlamaIndex imports
from llama_index.core import Document
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    PptxReader,
    MarkdownReader,
    # TxtReader,
)

# For tokenization
try:
    import tiktoken

    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

# For configuration
from pydantic import BaseModel, Field

# For contrastive learning (if available)
try:
    from sentence_transformers import SentenceTransformer, losses
    import torch
    from torch import nn
    import torch.nn.functional as F

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# File type constants
SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF",
    ".docx": "Word",
    ".doc": "Word",
    ".pptx": "PowerPoint",
    ".ppt": "PowerPoint",
    ".md": "Markdown",
    # ".txt": "Text",
}


class ClusteringConfig(BaseModel):
    """Configuration for file clustering."""

    input_dir: str = Field(..., description="Directory containing files to cluster")
    output_file: str = Field(..., description="Path to output JSON file")
    api_key: str = Field(..., description="Google Gemini API key")
    max_pages: int = Field(
        10, description="Maximum number of pages to process per document"
    )
    chunk_size: int = Field(
        1500, description="Size of chunks in tokens for sliding window"
    )
    chunk_overlap: int = Field(300, description="Overlap between chunks in tokens")
    use_contrastive_learning: bool = Field(
        True, description="Whether to use contrastive learning"
    )
    min_local_clusters: int = Field(2, description="Minimum number of local clusters")
    max_local_clusters: int = Field(20, description="Maximum number of local clusters")
    min_global_clusters: int = Field(2, description="Minimum number of global clusters")
    max_global_clusters: int = Field(
        15, description="Maximum number of global clusters"
    )
    contrastive_model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Sentence transformer model for contrastive learning",
    )
    temperature: float = Field(0.07, description="Temperature for contrastive loss")


class DocumentChunk:
    """Represents a chunk of a document."""

    def __init__(self, text: str, source_file: str, chunk_id: int):
        self.text = text
        self.source_file = source_file
        self.chunk_id = chunk_id
        self.embedding = None


class ContrastiveLearningModule:
    """Module for contrastive learning to enhance embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", temperature: float = 0.07):
        """
        Initialize the contrastive learning module.

        Args:
            model_name: Name of the sentence transformer model
            temperature: Temperature parameter for contrastive loss
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Sentence Transformers not available. Contrastive learning will be disabled."
            )
            self.available = False
            return

        self.available = True
        self.temperature = temperature
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized contrastive learning with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {e}")
            self.available = False

    def _create_augmented_view(self, text: str) -> str:
        """
        Create an augmented view of the text by applying simple transformations.

        Args:
            text: Original text

        Returns:
            Augmented text
        """
        if not self.available:
            return text

        # Simple augmentation: remove some random words (15%)
        words = text.split()
        if len(words) <= 5:
            return text

        import random

        keep_prob = 0.85
        augmented_words = [word for word in words if random.random() < keep_prob]

        # Ensure we don't remove too many words
        if len(augmented_words) < len(words) * 0.7:
            augmented_words = words

        return " ".join(augmented_words)

    def get_embeddings(
        self, texts: List[str], use_contrastive: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for texts, optionally using contrastive learning.

        Args:
            texts: List of texts to embed
            use_contrastive: Whether to use contrastive learning

        Returns:
            Array of embeddings
        """
        if not self.available or not use_contrastive:
            logger.warning("Contrastive learning not available, returning None")
            return None

        try:
            # Create augmented views
            augmented_texts = [self._create_augmented_view(text) for text in texts]

            # Get embeddings for original and augmented texts
            embeddings_original = self.model.encode(texts, convert_to_numpy=True)
            embeddings_augmented = self.model.encode(
                augmented_texts, convert_to_numpy=True
            )

            # Average the embeddings (simple approach)
            embeddings = (embeddings_original + embeddings_augmented) / 2.0

            return embeddings
        except Exception as e:
            logger.error(f"Error in contrastive learning: {e}")
            return None


class FileClusterer:
    """Class to cluster files based on their content using Google Gemini with hierarchical approach."""

    def __init__(self, config: ClusteringConfig):
        """
        Initialize with configuration.

        Args:
            config: Clustering configuration
        """
        self.config = config

        # Initialize document readers
        self.pdf_reader = PDFReader()
        self.docx_reader = DocxReader()
        self.pptx_reader = PptxReader()
        self.md_reader = MarkdownReader()
        # self.txt_reader = TxtReader()

        # Initialize Gemini API
        genai.configure(api_key=config.api_key)

        # Get embedding model
        self.embedding_model = genai.GenerativeModel("embedding-001")

        # Initialize contrastive learning module if available
        self.contrastive_module = ContrastiveLearningModule(
            model_name=config.contrastive_model_name, temperature=config.temperature
        )

        # Initialize tokenizer for chunking
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding(
                    "cl100k_base"
                )  # OpenAI's encoding
                logger.info("Using tiktoken for tokenization")
            except:
                self.tokenizer = None
                logger.warning(
                    "Failed to initialize tiktoken, using approximate tokenization"
                )
        else:
            self.tokenizer = None
            logger.warning("Tiktoken not available, using approximate tokenization")

        # Storage for document chunks and embeddings
        self.document_chunks = []
        self.local_clusters = {}
        self.global_clusters = {}

    def _is_supported_file(self, file_path: str) -> bool:
        """Check if the file type is supported."""
        _, ext = os.path.splitext(file_path.lower())
        return ext in SUPPORTED_EXTENSIONS

    def _get_file_type(self, file_path: str) -> str:
        """Get the type of file based on extension."""
        _, ext = os.path.splitext(file_path.lower())
        return SUPPORTED_EXTENSIONS.get(ext, "Unknown")

    def _extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from a file based on its type.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content
        """
        try:
            file_type = self._get_file_type(file_path)

            if file_type == "PDF":
                docs = self.pdf_reader.load_data(file_path)
            elif file_type == "Word":
                docs = self.docx_reader.load(file_path)
            elif file_type == "PowerPoint":
                docs = self.pptx_reader.load(file_path)
            elif file_type == "Markdown":
                docs = self.md_reader.load(file_path)
            # elif file_type == "Text":
            #     docs = self.txt_reader.load(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_type} for {file_path}")
                return ""

            # Combine all document texts
            if isinstance(docs, list):
                text = "\n\n".join([doc.text for doc in docs])
            else:
                text = docs.text

            return text
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            return ""

    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Approximate tokenization (rough estimate)
            return len(re.findall(r"\w+", text)) + len(re.findall(r"[^\w\s]", text))

    def _chunk_text(self, text: str, source_file: str) -> List[DocumentChunk]:
        """
        Split text into chunks using sliding window approach.

        Args:
            text: Text to chunk
            source_file: Source file path

        Returns:
            List of document chunks
        """
        if not text:
            return []

        chunks = []

        if self.tokenizer:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Create chunks with sliding window
            chunk_id = 0
            for i in range(
                0, len(tokens), self.config.chunk_size - self.config.chunk_overlap
            ):
                chunk_tokens = tokens[i : i + self.config.chunk_size]
                if len(chunk_tokens) < 50:  # Skip very small chunks
                    continue

                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(DocumentChunk(chunk_text, source_file, chunk_id))
                chunk_id += 1
        else:
            # Approximate chunking by paragraphs
            paragraphs = text.split("\n\n")
            current_chunk = []
            current_token_count = 0
            chunk_id = 0

            for para in paragraphs:
                para_token_count = self._count_tokens(para)

                if current_token_count + para_token_count > self.config.chunk_size:
                    # Current chunk is full, create a new one
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        chunks.append(DocumentChunk(chunk_text, source_file, chunk_id))
                        chunk_id += 1

                        # Start new chunk with overlap
                        overlap_size = min(
                            self.config.chunk_overlap, len(current_chunk)
                        )
                        current_chunk = current_chunk[-overlap_size:]
                        current_token_count = sum(
                            self._count_tokens(p) for p in current_chunk
                        )

                current_chunk.append(para)
                current_token_count += para_token_count

            # Add the last chunk if it's not empty
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(DocumentChunk(chunk_text, source_file, chunk_id))

        logger.info(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using Google Gemini.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if not text:
            logger.warning("Empty text provided for embedding")
            raise RuntimeError("Cannot generate embedding for empty text")

        try:
            # Truncate text if it's too long (Gemini has token limits)
            max_chars = 10000  # Adjust based on Gemini's limits
            if len(text) > max_chars:
                text = text[:max_chars]

            # Get embedding
            embedding = self.embedding_model.embed_content(
                content=text, task_type="retrieval_document"
            )

            return embedding.values
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Typical embedding dimension

    def _determine_optimal_clusters(
        self, embeddings: np.ndarray, min_clusters: int, max_clusters: int
    ) -> int:
        """
        Determine the optimal number of clusters using silhouette score.

        Args:
            embeddings: Document embeddings
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        # Adjust max_clusters based on number of samples
        n_samples = embeddings.shape[0]
        max_clusters = min(max_clusters, n_samples - 1)

        if max_clusters <= min_clusters:
            return max(2, min_clusters)

        best_score = -1
        best_n_clusters = min_clusters

        for n_clusters in range(min_clusters, max_clusters + 1):
            # Skip if we have too few samples for the number of clusters
            if n_samples <= n_clusters:
                continue

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # Calculate silhouette score
            try:
                score = silhouette_score(embeddings, cluster_labels)

                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except Exception as e:
                logger.warning(
                    f"Error calculating silhouette score for {n_clusters} clusters: {e}"
                )
                continue

        # If we have a very small number of documents, use a smaller number of clusters
        if n_samples < 10:
            best_n_clusters = max(2, n_samples // 2)

        logger.info(
            f"Optimal number of clusters: {best_n_clusters} (silhouette score: {best_score:.4f})"
        )
        return best_n_clusters

    def _generate_cluster_names(
        self,
        clusters: Dict[int, List[Union[str, DocumentChunk]]],
        is_chunk: bool = False,
    ) -> Dict[int, str]:
        """
        Generate descriptive names for each cluster using Gemini.

        Args:
            clusters: Dictionary mapping cluster IDs to lists of file paths or chunks
            is_chunk: Whether the clusters contain chunks or file paths

        Returns:
            Dictionary mapping cluster IDs to cluster names
        """
        cluster_names = {}

        # Create a generative model for text generation
        generation_model = genai.GenerativeModel("gemini-1.5-flash")

        for cluster_id, items in clusters.items():
            # Get sample texts from the cluster (up to 5 items)
            sample_items = items[:5]
            sample_texts = []

            for item in sample_items:
                if is_chunk:
                    # Item is a DocumentChunk
                    chunk = item
                    snippet = (
                        chunk.text[:500] + "..."
                        if len(chunk.text) > 500
                        else chunk.text
                    )
                    sample_texts.append(
                        f"Document: {os.path.basename(chunk.source_file)}, Chunk {chunk.chunk_id}\nContent snippet: {snippet}"
                    )
                else:
                    # Item is a file path
                    file_path = item
                    # Get a snippet of the content (first 500 chars)
                    content = self._extract_text_from_file(file_path)
                    snippet = content[:500] + "..." if len(content) > 500 else content
                    sample_texts.append(
                        f"File: {os.path.basename(file_path)}\nContent snippet: {snippet}"
                    )

            # Join sample texts
            samples = "\n\n".join(sample_texts)

            # Generate a cluster name using Gemini
            prompt = f"""
            I have a cluster of {'document chunks' if is_chunk else 'documents'} with the following content snippets:
            
            {samples}
            
            Based on these {'chunks' if is_chunk else 'documents'}, generate a short, descriptive category name (2-5 words) that captures the main theme or topic of this cluster.
            Respond with ONLY the category name, nothing else.
            """

            try:
                response = generation_model.generate_content(prompt)
                cluster_name = response.text.strip()

                # Fallback if the response is empty or too long
                if not cluster_name or len(cluster_name) > 50:
                    cluster_name = f"Category {cluster_id + 1}"

                cluster_names[cluster_id] = cluster_name

                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
            except Exception as e:
                logger.error(
                    f"Error generating cluster name for cluster {cluster_id}: {e}"
                )
                cluster_names[cluster_id] = f"Category {cluster_id + 1}"

        return cluster_names

    def cluster_files(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Cluster files using a hierarchical approach with sliding window chunking.

        Returns:
            Dictionary mapping global category names to dictionaries of local categories and file paths
        """
        # Find all supported files
        all_files = []
        for root, _, files in os.walk(self.config.input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_supported_file(file_path):
                    all_files.append(file_path)

        logger.info(f"Found {len(all_files)} supported files")

        if not all_files:
            logger.warning("No supported files found")
            return {}

        # Step 1: Extract text and create chunks
        logger.info("Extracting text and creating chunks...")
        for file_path in tqdm(all_files, desc="Processing files"):
            text = self._extract_text_from_file(file_path)
            if text:
                chunks = self._chunk_text(text, file_path)
                self.document_chunks.extend(chunks)

        logger.info(
            f"Created {len(self.document_chunks)} chunks from {len(all_files)} files"
        )

        if not self.document_chunks:
            logger.warning("No valid chunks created")
            return {}

        # Step 2: Get embeddings for chunks
        logger.info("Generating embeddings for chunks...")
        chunk_texts = [chunk.text for chunk in self.document_chunks]

        # Try to use contrastive learning if available
        contrastive_embeddings = None
        if self.config.use_contrastive_learning and self.contrastive_module.available:
            logger.info("Using contrastive learning for embeddings...")
            contrastive_embeddings = self.contrastive_module.get_embeddings(chunk_texts)

        if contrastive_embeddings is not None:
            logger.info("Successfully generated contrastive embeddings")
            for i, chunk in enumerate(self.document_chunks):
                chunk.embedding = contrastive_embeddings[i]
        else:
            logger.info("Using standard Gemini embeddings")
            for i, chunk in enumerate(
                tqdm(self.document_chunks, desc="Generating embeddings")
            ):
                chunk.embedding = self._get_embedding(chunk.text)

        # Step 3: Perform local clustering on chunks
        logger.info("Performing local clustering on chunks...")
        chunk_embeddings = np.array([chunk.embedding for chunk in self.document_chunks])

        n_local_clusters = self._determine_optimal_clusters(
            chunk_embeddings,
            min_clusters=self.config.min_local_clusters,
            max_clusters=self.config.max_local_clusters,
        )

        kmeans_local = KMeans(n_clusters=n_local_clusters, random_state=42, n_init=10)
        local_cluster_labels = kmeans_local.fit_predict(chunk_embeddings)

        # Group chunks by local cluster
        local_clusters = defaultdict(list)
        for i, chunk in enumerate(self.document_chunks):
            cluster_id = local_cluster_labels[i]
            local_clusters[cluster_id].append(chunk)

        # Generate local cluster names
        logger.info("Generating local cluster names...")
        local_cluster_names = self._generate_cluster_names(
            local_clusters, is_chunk=True
        )

        # Step 4: Create local cluster embeddings
        logger.info("Creating local cluster embeddings...")
        local_cluster_embeddings = []
        local_cluster_ids = []

        for cluster_id, chunks in local_clusters.items():
            # Average the embeddings of chunks in the cluster
            cluster_embedding = np.mean([chunk.embedding for chunk in chunks], axis=0)
            local_cluster_embeddings.append(cluster_embedding)
            local_cluster_ids.append(cluster_id)

        local_cluster_embeddings = np.array(local_cluster_embeddings)

        # Step 5: Perform global clustering on local clusters
        logger.info("Performing global clustering on local clusters...")
        n_global_clusters = self._determine_optimal_clusters(
            local_cluster_embeddings,
            min_clusters=self.config.min_global_clusters,
            max_clusters=self.config.max_global_clusters,
        )

        kmeans_global = KMeans(n_clusters=n_global_clusters, random_state=42, n_init=10)
        global_cluster_labels = kmeans_global.fit_predict(local_cluster_embeddings)

        # Group local clusters by global cluster
        global_clusters = defaultdict(list)
        for i, local_id in enumerate(local_cluster_ids):
            global_id = global_cluster_labels[i]
            global_clusters[global_id].append(local_id)

        # Step 6: Generate global cluster names
        logger.info("Generating global cluster names...")

        # Create a representation of global clusters for naming
        global_cluster_items = {}
        for global_id, local_ids in global_clusters.items():
            # Get a sample of chunks from each local cluster
            sample_chunks = []
            for local_id in local_ids:
                chunks = local_clusters[local_id]
                sample_chunks.extend(
                    chunks[:2]
                )  # Take up to 2 chunks from each local cluster
            global_cluster_items[global_id] = sample_chunks[
                :5
            ]  # Limit to 5 chunks for naming

        global_cluster_names = self._generate_cluster_names(
            global_cluster_items, is_chunk=True
        )

        # Step 7: Create the final hierarchical result
        logger.info("Creating final hierarchical result...")
        result = {}

        for global_id, local_ids in global_clusters.items():
            global_name = global_cluster_names[global_id]
            result[global_name] = {}

            for local_id in local_ids:
                local_name = local_cluster_names[local_id]

                # Get unique files in this local cluster
                file_paths = set()
                for chunk in local_clusters[local_id]:
                    file_paths.add(chunk.source_file)

                result[global_name][local_name] = list(file_paths)

        return result


def main():
    """Main function to run the script."""
    # Example configuration
    config = ClusteringConfig(
        input_dir="/path/to/input",
        output_file="/path/to/output.json",
        api_key="AIzaSyDP3Xx1qHgZEWE4Wk23ixCn2m5dAgogEDA",
    )

    # Override with command line arguments if provided
    import sys

    if len(sys.argv) > 1:
        config.input_dir = sys.argv[1]
    if len(sys.argv) > 2:
        config.output_file = sys.argv[2]
    # if len(sys.argv) > 3:
    #     config.api_key = sys.argv[3]

    # Create clusterer and cluster files
    clusterer = FileClusterer(config)
    result = clusterer.cluster_files()

    # Save result to JSON file
    with open(config.output_file, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    print(f"\nClustering complete! Results saved to {config.output_file}")
    print(f"Found {len(result)} global categories:")
    for global_category, local_categories in result.items():
        print(f"  {global_category}: {len(local_categories)} local categories")
        for local_category, files in local_categories.items():
            print(f"    - {local_category}: {len(files)} files")


if __name__ == "__main__":
    main()
