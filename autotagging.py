import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

class AutoTagger:
    def __init__(self, categories: Dict[str, Dict[str, List[str]]], threshold: float = 0.3):
        """
        Initialize AutoTagger with hierarchical categories and their keywords
        
        Args:
            categories: Dictionary mapping category names to their subcategories and keywords
            threshold: Minimum similarity score to assign a tag
        """
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.categories = categories
        
        # Create embeddings for each category and subcategory
        self.category_embeddings = {}
        for category, cat_data in categories.items():
            # Combine all keywords and descriptions for main category
            main_text = ' '.join(cat_data['keywords'])
            if 'description' in cat_data:
                main_text = cat_data['description'] + ' ' + main_text
            self.category_embeddings[category] = self.model.encode([main_text])[0]
            
            # Handle subcategories if they exist
            if 'subcategories' in cat_data:
                for subcat, subcat_data in cat_data['subcategories'].items():
                    subcat_text = ' '.join(subcat_data['keywords'])
                    if 'description' in subcat_data:
                        subcat_text = subcat_data['description'] + ' ' + subcat_text
                    self.category_embeddings[f"{category}/{subcat}"] = self.model.encode([subcat_text])[0]
    
    def generate_tags(self, text: str) -> List[Tuple[str, float]]:
        """Generate tags with improved confidence scoring"""
        text_embedding = self.model.encode([text])[0]
        
        tags = []
        for category, category_embedding in self.category_embeddings.items():
            similarity = cosine_similarity(
                [text_embedding], 
                [category_embedding]
            )[0][0]
            
            if similarity >= self.threshold:
                tags.append((category, float(similarity)))
        
        return sorted(tags, key=lambda x: x[1], reverse=True)

# Example usage
if __name__ == "__main__":
    # Enhanced categories with descriptions and subcategories
    categories = {
        "technology": {
            "description": "Topics related to digital technology, computing, and technical innovation",
            "keywords": ["computer", "software", "hardware", "digital", "internet", "tech", "AI", "programming",
                        "update", "security", "system", "application", "development", "code", "algorithm",
                        "database", "network", "cloud", "cybersecurity", "encryption"],
            "subcategories": {
                "software": {
                    "description": "Software development and applications",
                    "keywords": ["application", "program", "update", "feature", "bug fix", "release",
                               "version", "patch", "security update", "software development"]
                },
                "ai": {
                    "description": "Artificial Intelligence and Machine Learning",
                    "keywords": ["artificial intelligence", "machine learning", "neural network", "deep learning",
                               "NLP", "computer vision", "AI model", "training", "inference", "algorithm"]
                }
            }
        },
        "finance": {
            "description": "Topics related to money, markets, and financial services",
            "keywords": ["money", "banking", "investment", "market", "trading", "financial", "economy",
                        "stock", "shares", "profit", "loss", "growth", "revenue", "assets", "fund",
                        "portfolio", "dividend", "broker", "exchange", "currency"],
            "subcategories": {
                "markets": {
                    "description": "Stock markets and trading",
                    "keywords": ["stock market", "trading", "shares", "bulls", "bears", "investment",
                               "market analysis", "market trend", "market growth", "market index"]
                }
            }
        },
        "healthcare": {
            "description": "Medical and health-related topics",
            "keywords": ["medical", "health", "doctor", "patient", "treatment", "hospital", "medicine",
                        "wellness", "diagnosis", "therapy", "healthcare", "clinical", "prescription",
                        "symptoms", "disease", "prevention", "care", "medical procedure", "rehabilitation",
                        "exercise", "fitness", "nutrition", "mental health"]
        }
    }

    # Initialize tagger
    tagger = AutoTagger(categories)

    # Test texts
    test_texts = [
        "The new software update includes improved security features",
        "The stock market showed significant growth today",
        "Doctors recommend regular exercise for better health",
        "The latest AI models are transforming how we process language",
        "New security patch released for the database system",
        "Investment strategies for volatile market conditions"
    ]

    # Generate and print tags
    print("\nGenerating tags for test texts:")
    print("-" * 50)
    for text in test_texts:
        tags = tagger.generate_tags(text)
        print(f"\nText: {text}")
        if tags:
            print("Tags:")
            for category, confidence in tags:
                print(f"- {category}: {confidence:.3f}")
        else:
            print("No tags found")