"""
Behavior Processor: Xử lý dữ liệu hành vi người dùng
(clicks, views, purchases, time_spent, cart_adds, etc.)
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List
import pickle

class BehaviorProcessor:
    """
    Xử lý behavior features
    
    Input: Raw behavior data (clicks, views, purchase history, etc.)
    Output: Normalized behavior vectors
    
    Features:
    - Click-through rate (CTR)
    - View duration
    - Purchase frequency
    - Cart add rate
    - Category preferences
    - Time-based patterns (hour of day, day of week)
    """
    
    def __init__(self, feature_dim: int = 50):
        """
        Args:
            feature_dim: Số lượng behavior features
        """
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def extract_features(self, user_data: pd.DataFrame) -> np.ndarray:
        """
        Trích xuất behavior features từ raw data
        
        Args:
            user_data: DataFrame chứa user interaction history
                Columns: user_id, item_id, action_type, timestamp, duration, etc.
                
        Returns:
            features: numpy array shape (num_users, feature_dim)
        """
        features_list = []
        
        for user_id in user_data['user_id'].unique():
            user_interactions = user_data[user_data['user_id'] == user_id]
            
            # Feature 1-10: Action counts
            num_clicks = len(user_interactions[user_interactions['action_type'] == 'click'])
            num_views = len(user_interactions[user_interactions['action_type'] == 'view'])
            num_purchases = len(user_interactions[user_interactions['action_type'] == 'purchase'])
            num_cart_adds = len(user_interactions[user_interactions['action_type'] == 'cart_add'])
            num_wishlist = len(user_interactions[user_interactions['action_type'] == 'wishlist'])
            
            # Feature 11-15: Rates
            total_interactions = len(user_interactions)
            ctr = num_clicks / max(total_interactions, 1)
            purchase_rate = num_purchases / max(num_views, 1)
            cart_conversion = num_purchases / max(num_cart_adds, 1)
            
            # Feature 16-20: Time-based
            avg_session_duration = user_interactions['duration'].mean() if 'duration' in user_interactions else 0
            avg_time_between_actions = user_interactions['timestamp'].diff().mean().total_seconds() if len(user_interactions) > 1 else 0
            
            # Feature 21-30: Category preferences (top 10 categories)
            category_counts = user_interactions['category'].value_counts().head(10) if 'category' in user_interactions else pd.Series()
            category_features = category_counts.values.tolist() + [0] * (10 - len(category_counts))
            
            # Feature 31-40: Recency features
            days_since_last_action = (pd.Timestamp.now() - user_interactions['timestamp'].max()).days if len(user_interactions) > 0 else 999
            days_since_last_purchase = (pd.Timestamp.now() - user_interactions[user_interactions['action_type'] == 'purchase']['timestamp'].max()).days if num_purchases > 0 else 999
            
            # Feature 41-50: Price sensitivity & other behavioral patterns
            avg_price_viewed = user_interactions['price'].mean() if 'price' in user_interactions else 0
            price_variance = user_interactions['price'].std() if 'price' in user_interactions else 0
            
            # Combine all features
            user_features = [
                num_clicks, num_views, num_purchases, num_cart_adds, num_wishlist,
                ctr, purchase_rate, cart_conversion, 0, 0,  # 10 features
                avg_session_duration, avg_time_between_actions, 0, 0, 0,  # 15 features
                *category_features,  # 25 features
                days_since_last_action, days_since_last_purchase, 0, 0, 0,  # 30 features
                0, 0, 0, 0, 0,  # 35 features
                avg_price_viewed, price_variance, 0, 0, 0,  # 40 features
                0, 0, 0, 0, 0,  # 45 features
                0, 0, 0, 0, 0   # 50 features
            ]
            
            features_list.append(user_features[:self.feature_dim])
        
        return np.array(features_list)
    
    def fit(self, features: np.ndarray):
        """
        Fit scaler trên training data
        
        Args:
            features: numpy array shape (num_samples, feature_dim)
        """
        self.scaler.fit(features)
        self.is_fitted = True
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features
        
        Args:
            features: numpy array shape (num_samples, feature_dim)
            
        Returns:
            normalized_features: numpy array cùng shape
        """
        if not self.is_fitted:
            raise ValueError("Scaler chưa được fit! Gọi .fit() trước.")
            
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit và transform cùng lúc"""
        self.fit(features)
        return self.transform(features)
    
    def generate_synthetic_data(self, num_users: int = 1000, 
                                num_items: int = 500) -> pd.DataFrame:
        """
        Tạo dữ liệu behavior giả lập để test
        
        Args:
            num_users: Số lượng users
            num_items: Số lượng items
            
        Returns:
            df: DataFrame chứa synthetic behavior data
        """
        np.random.seed(42)
        
        data = []
        action_types = ['click', 'view', 'purchase', 'cart_add', 'wishlist']
        categories = ['electronics', 'fashion', 'home', 'sports', 'books', 
                     'food', 'toys', 'beauty', 'automotive', 'health']
        
        for _ in range(num_users * 50):  # Mỗi user có ~50 interactions
            data.append({
                'user_id': np.random.randint(0, num_users),
                'item_id': np.random.randint(0, num_items),
                'action_type': np.random.choice(action_types, p=[0.4, 0.3, 0.1, 0.15, 0.05]),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
                'duration': np.random.exponential(scale=30),  # seconds
                'category': np.random.choice(categories),
                'price': np.random.uniform(10, 500)
            })
        
        return pd.DataFrame(data)
    
    def save_scaler(self, filepath: str):
        """Lưu fitted scaler"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, filepath: str):
        """Load fitted scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        self.is_fitted = True


# Example usage
if __name__ == "__main__":
    processor = BehaviorProcessor(feature_dim=50)
    
    # Generate synthetic data
    df = processor.generate_synthetic_data(num_users=100, num_items=500)
    print(f"Generated {len(df)} interactions for 100 users")
    
    # Extract features
    features = processor.extract_features(df)
    print(f"Behavior features shape: {features.shape}")
    
    # Normalize
    normalized_features = processor.fit_transform(features)
    print(f"Normalized features shape: {normalized_features.shape}")
    print(f"Sample features (first 5 dims): {normalized_features[0][:5]}")