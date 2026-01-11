"""
Synthetic Data Generator for Multi-Modal Recommendation System
Generates users, items, and interactions with multi-modal features
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta
import random
from tqdm import tqdm

class SyntheticDataGenerator:
    def __init__(
        self,
        num_users: int = 1000,
        num_items: int = 10000,
        num_interactions: int = 50000,
        seed: int = 42
    ):
        """
        Initialize synthetic data generator
        
        Args:
            num_users: Number of users to generate
            num_items: Number of items to generate
            num_interactions: Number of user-item interactions
            seed: Random seed for reproducibility
        """
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Item categories
        self.categories = [
            'Electronics', 'Books', 'Clothing', 'Home & Garden',
            'Sports', 'Toys', 'Food', 'Beauty', 'Automotive', 'Health'
        ]
        
        # User preference types
        self.preference_types = [
            'text_heavy',      # Reads descriptions carefully
            'image_heavy',     # Decides based on images
            'behavior_heavy',  # Influenced by ratings/popularity
            'balanced'         # Uses all modalities equally
        ]
        
    def generate_items(self) -> pd.DataFrame:
        """Generate synthetic items with multi-modal features"""
        print("Generating items...")
        
        items = []
        for item_id in tqdm(range(self.num_items)):
            category = np.random.choice(self.categories)
            
            # Generate text features (keywords representation)
            text_keywords = self._generate_text_features(category)
            
            # Generate image features (simulated visual characteristics)
            image_features = self._generate_image_features(category)
            
            # Generate behavior features (popularity, ratings)
            behavior_features = self._generate_behavior_features()
            
            item = {
                'item_id': item_id,
                'category': category,
                'name': f"{category}_Item_{item_id}",
                'text_keywords': text_keywords,
                'image_features': image_features,
                'avg_rating': behavior_features['avg_rating'],
                'num_ratings': behavior_features['num_ratings'],
                'popularity_score': behavior_features['popularity_score'],
                'price': np.random.uniform(10, 1000),
                'brand': f"Brand_{np.random.randint(0, 50)}"
            }
            items.append(item)
        
        return pd.DataFrame(items)
    
    def generate_users(self) -> pd.DataFrame:
        """Generate synthetic users with preference types"""
        print("Generating users...")
        
        users = []
        for user_id in tqdm(range(self.num_users)):
            # Assign preference type with specified distribution
            preference_type = np.random.choice(
                self.preference_types,
                p=[0.3, 0.3, 0.2, 0.2]  # From config
            )
            
            # Generate user demographics
            age = np.random.randint(18, 70)
            
            # Category preferences
            preferred_categories = np.random.choice(
                self.categories,
                size=np.random.randint(2, 5),
                replace=False
            ).tolist()
            
            user = {
                'user_id': user_id,
                'preference_type': preference_type,
                'age': age,
                'preferred_categories': preferred_categories,
                'registration_date': self._random_date()
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_interactions(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate user-item interactions based on user preferences"""
        print("Generating interactions...")
        
        interactions = []
        
        for _ in tqdm(range(self.num_interactions)):
            user = users_df.sample(1).iloc[0]
            
            # Select item based on user preference type
            item = self._select_item_for_user(user, items_df)
            
            # Generate interaction features
            interaction = {
                'user_id': user['user_id'],
                'item_id': item['item_id'],
                'rating': self._generate_rating(user, item),
                'timestamp': self._random_timestamp(),
                'interaction_type': np.random.choice(
                    ['view', 'click', 'purchase', 'rate'],
                    p=[0.5, 0.3, 0.15, 0.05]
                ),
                'session_duration': np.random.exponential(120),  # seconds
                'device': np.random.choice(['mobile', 'desktop', 'tablet'])
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def _generate_text_features(self, category: str) -> List[str]:
        """Generate text keywords for item"""
        # Simulate different keyword distributions per category
        category_keywords = {
            'Electronics': ['powerful', 'fast', 'efficient', 'modern', 'wireless'],
            'Books': ['interesting', 'educational', 'engaging', 'bestseller', 'classic'],
            'Clothing': ['stylish', 'comfortable', 'fashionable', 'quality', 'elegant'],
            'Home & Garden': ['durable', 'practical', 'beautiful', 'functional', 'eco-friendly'],
            'Sports': ['professional', 'lightweight', 'strong', 'performance', 'athletic'],
            'Toys': ['fun', 'educational', 'safe', 'creative', 'colorful'],
            'Food': ['delicious', 'healthy', 'organic', 'fresh', 'gourmet'],
            'Beauty': ['natural', 'effective', 'gentle', 'premium', 'luxurious'],
            'Automotive': ['reliable', 'powerful', 'safe', 'efficient', 'advanced'],
            'Health': ['effective', 'safe', 'natural', 'certified', 'trusted']
        }
        
        keywords = category_keywords.get(category, ['quality', 'great', 'recommended'])
        return np.random.choice(keywords, size=np.random.randint(2, 4), replace=False).tolist()
    
    def _generate_image_features(self, category: str) -> Dict[str, float]:
        """Generate simulated image features"""
        # Simulate visual characteristics
        return {
            'brightness': np.random.uniform(0.3, 0.9),
            'contrast': np.random.uniform(0.4, 0.8),
            'color_variance': np.random.uniform(0.2, 0.7),
            'sharpness': np.random.uniform(0.5, 1.0)
        }
    
    def _generate_behavior_features(self) -> Dict[str, float]:

        num_ratings = int(np.random.exponential(50))
    
    # Use uniform distribution instead of beta to avoid skew
        avg_rating = np.random.uniform(2.0, 4.5)  # Reasonable range
    
        popularity_score = np.log1p(num_ratings) * avg_rating / 5
    
        return {
            'num_ratings': num_ratings,
            'avg_rating': round(avg_rating, 2),
            'popularity_score': round(popularity_score, 2)
        }
    
    def _select_item_for_user(
        self,
        user: pd.Series,
        items_df: pd.DataFrame
    ) -> pd.Series:
        """Select item based on user's preference type"""
        # Filter items by user's preferred categories (with some randomness)
        if np.random.random() < 0.7:
            filtered_items = items_df[
                items_df['category'].isin(user['preferred_categories'])
            ]
            if len(filtered_items) == 0:
                filtered_items = items_df
        else:
            filtered_items = items_df
        
        # Adjust selection probability based on preference type
        preference_type = user['preference_type']
        
        if preference_type == 'behavior_heavy':
            # Prefer popular items
            weights = filtered_items['popularity_score'].values
            weights = weights / weights.sum()
            idx = np.random.choice(len(filtered_items), p=weights)
            return filtered_items.iloc[idx]
        
        elif preference_type == 'image_heavy':
            # Random selection (image features are simulated uniformly)
            return filtered_items.sample(1).iloc[0]
        
        elif preference_type == 'text_heavy':
            # Random selection (text features affect rating generation)
            return filtered_items.sample(1).iloc[0]
        
        else:  # balanced
            return filtered_items.sample(1).iloc[0]
    
    def _generate_rating(self, user: pd.Series, item: pd.Series) -> int:
    
    # Base distribution: balanced with slight realistic skew
    # 10% / 15% / 25% / 30% / 20% = reasonable real-world distribution
        base_probs = {
            'text_heavy':     [0.15, 0.20, 0.30, 0.25, 0.10],  # More critical
            'image_heavy':    [0.10, 0.15, 0.25, 0.30, 0.20],  # Balanced
            'behavior_heavy': [0.08, 0.12, 0.20, 0.35, 0.25],  # Follow positive trend
            'balanced':       [0.10, 0.15, 0.25, 0.30, 0.20]   # Realistic baseline
        }
    
        preference_type = user['preference_type']
        probs = base_probs.get(preference_type, base_probs['balanced'])
    
    # Generate rating
        rating = np.random.choice([1, 2, 3, 4, 5], p=probs)
    
        return int(rating)
    
    def _random_date(self) -> str:
        """Generate random registration date"""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        time_delta = end_date - start_date
        random_days = np.random.randint(0, time_delta.days)
        
        return (start_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def _random_timestamp(self) -> str:
        """Generate random interaction timestamp"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        time_delta = end_date - start_date
        random_seconds = np.random.randint(0, int(time_delta.total_seconds()))
        
        return (start_date + timedelta(seconds=random_seconds)).strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_all(self, output_dir: str = './data/raw') -> Dict[str, pd.DataFrame]:
        """Generate all synthetic data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n=== Generating Synthetic Dataset ===")
        
        # Generate data
        items_df = self.generate_items()
        users_df = self.generate_users()
        interactions_df = self.generate_interactions(users_df, items_df)
        
        # Save to CSV
        print("\nSaving data...")
        items_df.to_csv(output_path / 'items.csv', index=False)
        users_df.to_csv(output_path / 'users.csv', index=False)
        interactions_df.to_csv(output_path / 'interactions.csv', index=False)
        
        # Save metadata
        metadata = {
            'num_users': self.num_users,
            'num_items': self.num_items,
            'num_interactions': self.num_interactions,
            'categories': self.categories,
            'preference_types': self.preference_types,
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Data saved to {output_path}")
        print(f"   - items.csv: {len(items_df)} items")
        print(f"   - users.csv: {len(users_df)} users")
        print(f"   - interactions.csv: {len(interactions_df)} interactions")
        
        return {
            'items': items_df,
            'users': users_df,
            'interactions': interactions_df
        }


if __name__ == "__main__":
    # Generate synthetic dataset
    generator = SyntheticDataGenerator(
        num_users=1000,
        num_items=10000,
        num_interactions=50000,
        seed=42
    )
    
    data = generator.generate_all()
    
    # Display statistics
    print("\n=== Dataset Statistics ===")
    print(f"Users by preference type:")
    print(data['users']['preference_type'].value_counts())
    print(f"\nItems by category:")
    print(data['items']['category'].value_counts())
    print(f"\nInteractions by type:")
    print(data['interactions']['interaction_type'].value_counts())