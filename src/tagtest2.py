from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import argparse
from tqdm.auto import tqdm
import json
from collections import defaultdict
import os

@dataclass
class TaggingResult:
    text: str
    categories: Dict[str, float]  # category -> confidence score
    tags: Dict[str, List[str]]    # category -> list of relevant tags
    metadata: Dict[str, Any]
    similarity_scores: Dict[str, Dict[str, float]]  # category -> tag -> score

class HierarchicalTagger:
    def __init__(self):
        """Initialize tagger with embedding model."""
        self.logger = self.setup_logger()
        
        try:
            model_dir = '/app/models/sentence-transformers_all-MiniLM-L6-v2'
            self.logger.info(f"Loading model from: {model_dir}")
            self.model = SentenceTransformer(model_dir)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
        self.batch_size = 32
        self.category_embeddings = {}
        self.tag_embeddings = {}
        self.taxonomy = {}

    @staticmethod
    def setup_logger():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('HierarchicalTagger')

    def load_taxonomy(self, taxonomy_path: str):
        """Load and process hierarchical taxonomy metadata."""
        try:
            # Handle directory paths
            if os.path.isdir(taxonomy_path):
                taxonomy_dir = taxonomy_path
                # Look for metadata.json in the directory
                taxonomy_path = os.path.join(taxonomy_dir, 'metadata.json')
            
            # Log the resolved path
            self.logger.info(f"Resolved taxonomy path: {taxonomy_path}")
            
            # Verify file exists
            if not os.path.exists(taxonomy_path):
                raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")
            
            # Load and validate taxonomy
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                taxonomy_data = json.load(f)
            
            self.taxonomy = taxonomy_data.get('categories', {})
            if not self.taxonomy:
                raise ValueError("No categories found in taxonomy")
            
            self.logger.info(f"Loaded taxonomy with {len(self.taxonomy)} categories")
            
            # Compute embeddings for categories using their descriptions
            for category, content in self.taxonomy.items():
                # Combine descriptions for rich category representation
                category_text = ' '.join(content.get('descriptions', []))
                self.category_embeddings[category] = self.model.encode(category_text)
                
                # Compute embeddings for individual tags
                self.tag_embeddings[category] = {}
                for tag in content.get('tags', []):
                    # Combine tag with category descriptions for context
                    tag_text = f"{tag} {category_text}"
                    self.tag_embeddings[category][tag] = self.model.encode(tag_text)
            
            self.logger.info("Computed embeddings for all categories and tags")
            
        except Exception as e:
            self.logger.error(f"Error loading taxonomy: {str(e)}")
            raise

    def compute_similarity(self, text_embedding: np.ndarray, 
                         reference_embedding: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        return float(np.dot(text_embedding, reference_embedding) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(reference_embedding)
        ))

    def assign_categories_and_tags(self, text: str, 
                                 category_threshold: float = 0.2,
                                 tag_threshold: float = 0.2) -> Tuple[Dict[str, float], Dict[str, List[str]], Dict[str, Dict[str, float]]]:
        """Assign categories and tags based on semantic similarity."""
        text_embedding = self.model.encode(text)
        
        # Calculate category similarities
        category_scores = {
            category: self.compute_similarity(text_embedding, category_emb)
            for category, category_emb in self.category_embeddings.items()
        }
        
        # Filter categories by threshold
        relevant_categories = {
            category: score
            for category, score in category_scores.items()
            if score > category_threshold
        }
        
        # For relevant categories, find matching tags
        relevant_tags = {}
        tag_similarities = {}
        
        for category in relevant_categories:
            tag_scores = {
                tag: self.compute_similarity(text_embedding, tag_emb)
                for tag, tag_emb in self.tag_embeddings[category].items()
            }
            
            # Store all tag similarities for metrics
            tag_similarities[category] = tag_scores
            
            # Filter and sort tags by threshold
            matched_tags = sorted([
                (tag, score) for tag, score in tag_scores.items()
                if score > tag_threshold
            ], key=lambda x: x[1], reverse=True)
            
            if matched_tags:
                # Take top 5 tags with highest similarity
                relevant_tags[category] = [tag for tag, _ in matched_tags[:5]]
        
        return relevant_categories, relevant_tags, tag_similarities

    def process_text(self, text: str) -> TaggingResult:
        """Process a single text input and generate hierarchical tags."""
        # Assign categories and tags
        categories, tags, similarities = self.assign_categories_and_tags(text)
        
        # Create metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'char_count': len(text),
            'word_count': len(text.split()),
            'category_confidence': max(categories.values()) if categories else 0.0,
            'processing_version': '1.0'
        }
        
        return TaggingResult(
            text=text,
            categories=categories,
            tags=tags,
            metadata=metadata,
            similarity_scores=similarities
        )

    def process_batch(self, texts: List[str]) -> List[TaggingResult]:
        """Process a batch of texts."""
        if not self.taxonomy:
            raise ValueError("Taxonomy must be loaded before processing")
            
        results = []
        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch = texts[i:i + self.batch_size]
            batch_results = [self.process_text(text) for text in batch]
            results.extend(batch_results)
        
        return results

    def save_results(self, results: List[TaggingResult], output_path: str, metrics_path: str):
        """Save tagging results and metrics."""
        output_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_documents': len(results),
                'version': '1.0'
            },
            'results': [
                {
                    'text': r.text,
                    'categories': r.categories,
                    'tags': r.tags,
                    'metadata': r.metadata,
                    'similarity_scores': r.similarity_scores
                }
                for r in results
            ]
        }
        
        # Calculate comprehensive metrics
        metrics = {
            'num_documents': len(results),
            'avg_categories_per_doc': sum(len(r.categories) for r in results) / len(results),
            'avg_tags_per_category': self.calculate_avg_tags_per_category(results),
            'category_distribution': self.calculate_category_distribution(results),
            'tag_distribution': self.calculate_tag_distribution(results),
            'category_correlations': self.calculate_category_correlations(results),
            'confidence_metrics': self.calculate_confidence_metrics(results)
        }
        
        # Ensure output directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        
        # Save files
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
            
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    def calculate_avg_tags_per_category(self, results: List[TaggingResult]) -> Dict[str, float]:
        """Calculate average number of tags per category."""
        category_tag_counts = defaultdict(list)
        
        for result in results:
            for category, tags in result.tags.items():
                category_tag_counts[category].append(len(tags))
        
        return {
            category: round(sum(counts) / len(counts), 2)
            for category, counts in category_tag_counts.items()
            if counts
        }

    def calculate_category_distribution(self, results: List[TaggingResult]) -> Dict[str, Dict[str, Any]]:
        """Calculate detailed distribution of categories."""
        distribution = defaultdict(lambda: {'count': 0, 'avg_confidence': 0.0})
        
        for result in results:
            for category, confidence in result.categories.items():
                distribution[category]['count'] += 1
                distribution[category]['avg_confidence'] += confidence
        
        # Calculate averages
        for stats in distribution.values():
            stats['avg_confidence'] = round(
                stats['avg_confidence'] / stats['count'], 3
            ) if stats['count'] > 0 else 0
        
        return dict(sorted(
            distribution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        ))

    def calculate_tag_distribution(self, results: List[TaggingResult]) -> Dict[str, Dict[str, int]]:
        """Calculate distribution of tags within each category."""
        distribution = defaultdict(lambda: defaultdict(int))
        
        for result in results:
            for category, tags in result.tags.items():
                for tag in tags:
                    distribution[category][tag] += 1
        
        return {
            category: dict(sorted(
                tags.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            for category, tags in distribution.items()
        }

    def calculate_category_correlations(self, results: List[TaggingResult]) -> Dict[str, Dict[str, float]]:
        """Calculate correlations between categories."""
        correlations = {}
        total_docs = len(results)
        
        for cat1 in self.taxonomy:
            correlations[cat1] = {}
            for cat2 in self.taxonomy:
                if cat1 != cat2:
                    cooccurrences = sum(
                        1 for r in results
                        if cat1 in r.categories and cat2 in r.categories
                    )
                    correlation = cooccurrences / total_docs if total_docs > 0 else 0
                    correlations[cat1][cat2] = round(correlation, 3)
        
        return correlations

    def calculate_confidence_metrics(self, results: List[TaggingResult]) -> Dict[str, Any]:
        """Calculate confidence metrics for categories and tags."""
        category_confidences = [
            score for r in results for score in r.categories.values()
        ]
        
        tag_confidences = [
            score for r in results 
            for category_scores in r.similarity_scores.values()
            for score in category_scores.values()
        ]
        
        return {
            'categories': {
                'avg_confidence': round(sum(category_confidences) / len(category_confidences), 3)
                if category_confidences else 0,
                'min_confidence': round(min(category_confidences), 3)
                if category_confidences else 0,
                'max_confidence': round(max(category_confidences), 3)
                if category_confidences else 0
            },
            'tags': {
                'avg_confidence': round(sum(tag_confidences) / len(tag_confidences), 3)
                if tag_confidences else 0,
                'min_confidence': round(min(tag_confidences), 3)
                if tag_confidences else 0,
                'max_confidence': round(max(tag_confidences), 3)
                if tag_confidences else 0
            }
        }

def load_input_data(data_path: str) -> List[str]:
    """Load input data with enhanced directory handling."""
    try:
        # Handle directory paths
        if os.path.isdir(data_path):
            data_dir = data_path
            # Look for output.json in the directory
            for file in os.listdir(data_dir):
                if file.endswith('.json'):
                    data_path = os.path.join(data_dir, file)
                    break
        
        # Log the resolved path
        logging.info(f"Resolved input data path: {data_path}")
        
        # Verify file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Input file not found: {data_path}")
            
        # Load and validate data
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, dict) or 'documents' not in data:
            raise ValueError("Invalid data format - missing 'documents' key")
            
        texts = [doc.get('text', '') for doc in data['documents']]
        if not texts:
            raise ValueError("No text documents found in data")
            
        logging.info(f"Successfully loaded {len(texts)} documents")
        return texts
        
    except Exception as e:
        logging.error(f"Error loading input data: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Hierarchical Semantic Tagger')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input data file or directory')
    parser.add_argument('--taxonomy', type=str, required=True,
                      help='Path to taxonomy metadata file or directory')
    parser.add_argument('--output', type=str, required=True,
                      help='Path for tagging results output')
    parser.add_argument('--metrics', type=str, required=True,
                      help='Path for metrics output')
    
    args = parser.parse_args()
    logger = logging.getLogger('HierarchicalTagger')
    
    try:
        # Initialize tagger
        logger.info("Initializing tagger...")
        tagger = HierarchicalTagger()
        
        # Load taxonomy
        logger.info("Loading taxonomy...")
        tagger.load_taxonomy(args.taxonomy)
        
        # Load input data
        logger.info("Loading input data...")
        texts = load_input_data(args.input)
        
        # Process texts
        logger.info("Processing texts...")
        results = tagger.process_batch(texts)
        
        # Save results and metrics
        logger.info("Saving results and metrics...")
        tagger.save_results(results, args.output, args.metrics)
        
        print("\nTagging completed successfully!")
        print(f"Results saved to: {args.output}")
        print(f"Metrics saved to: {args.metrics}")
        
    except Exception as e:
        logger.error(f"Tagging failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()