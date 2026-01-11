"""
Non-IID Data Splitter using Dirichlet Distribution
Splits users and their interactions across federated clients
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class NonIIDDataSplitter:
    def __init__(
        self,
        num_clients: int = 10,
        alpha: float = 0.5,
        seed: int = 42
    ):
        """
        Initialize Non-IID data splitter
        
        Args:
            num_clients: Number of federated clients
            alpha: Dirichlet concentration parameter
                   - Lower alpha (e.g., 0.1) = More non-IID
                   - Higher alpha (e.g., 10.0) = More IID
            seed: Random seed
        """
        self.num_clients = num_clients
        self.alpha = alpha
        self.seed = seed
        
        np.random.seed(seed)
        
    def split_by_dirichlet(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> Dict[int, Dict]:
        """
        Split data using Dirichlet distribution
        
        Strategy:
        1. Group users by preference_type
        2. Use Dirichlet to assign users to clients (non-uniform)
        3. Each client gets their users' interactions
        """
        print("\n=== Splitting Data with Dirichlet Distribution ===")
        print(f"Alpha: {self.alpha} (lower = more non-IID)")
        
        # Group users by preference type
        preference_types = users_df['preference_type'].unique()
        
        # Initialize client data
        client_data = {
            client_id: {
                'users': [],
                'interactions': [],
                'preference_distribution': defaultdict(int),
                'category_distribution': defaultdict(int)
            }
            for client_id in range(self.num_clients)
        }
        
        # For each preference type, use Dirichlet to distribute users
        for pref_type in preference_types:
            users_with_pref = users_df[users_df['preference_type'] == pref_type]
            user_ids = users_with_pref['user_id'].values
            
            # Generate Dirichlet distribution
            proportions = np.random.dirichlet(
                [self.alpha] * self.num_clients
            )
            
            # Calculate number of users per client
            users_per_client = (proportions * len(user_ids)).astype(int)
            
            # Adjust for rounding errors
            users_per_client[-1] += len(user_ids) - users_per_client.sum()
            
            # Shuffle and split users
            np.random.shuffle(user_ids)
            start_idx = 0
            
            for client_id, num_users in enumerate(users_per_client):
                if num_users > 0:
                    assigned_users = user_ids[start_idx:start_idx + num_users]
                    client_data[client_id]['users'].extend(assigned_users.tolist())
                    client_data[client_id]['preference_distribution'][pref_type] += num_users
                    start_idx += num_users
        
        # Assign interactions to clients based on users
        print("\nAssigning interactions to clients...")
        for client_id in range(self.num_clients):
            client_users = set(client_data[client_id]['users'])
            
            # Get interactions for these users
            client_interactions = interactions_df[
                interactions_df['user_id'].isin(client_users)
            ]
            
            client_data[client_id]['interactions'] = client_interactions
            
            # Calculate category distribution
            interacted_items = items_df[
                items_df['item_id'].isin(client_interactions['item_id'])
            ]
            
            for category in interacted_items['category']:
                client_data[client_id]['category_distribution'][category] += 1
        
        # Print statistics
        self._print_distribution_stats(client_data, users_df)
        
        return client_data
    
    def split_by_user_clustering(
        self,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        interactions_df: pd.DataFrame
    ) -> Dict[int, Dict]:
        """
        Alternative: Split by clustering similar users together
        Creates more realistic client groupings
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.cluster import KMeans
        
        print("\n=== Clustering-based Split ===")
        
        # Encode categorical features
        le_pref = LabelEncoder()
        users_df['pref_encoded'] = le_pref.fit_transform(users_df['preference_type'])
        
        # Create feature matrix
        features = []
        for _, user in users_df.iterrows():
            feat = [
                user['pref_encoded'],
                user['age'] / 100,  # Normalize
                len(user['preferred_categories']) / 10
            ]
            features.append(feat)
        
        features = np.array(features)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=self.num_clients, random_state=self.seed)
        users_df['cluster'] = kmeans.fit_predict(features)
        
        # Assign users to clients based on clusters
        client_data = {
            client_id: {
                'users': [],
                'interactions': [],
                'preference_distribution': defaultdict(int),
                'category_distribution': defaultdict(int)
            }
            for client_id in range(self.num_clients)
        }
        
        for client_id in range(self.num_clients):
            client_users_df = users_df[users_df['cluster'] == client_id]
            client_data[client_id]['users'] = client_users_df['user_id'].tolist()
            
            # Count preferences
            for pref in client_users_df['preference_type']:
                client_data[client_id]['preference_distribution'][pref] += 1
            
            # Get interactions
            client_user_ids = set(client_data[client_id]['users'])
            client_interactions = interactions_df[
                interactions_df['user_id'].isin(client_user_ids)
            ]
            client_data[client_id]['interactions'] = client_interactions
            
            # Category distribution
            interacted_items = items_df[
                items_df['item_id'].isin(client_interactions['item_id'])
            ]
            for category in interacted_items['category']:
                client_data[client_id]['category_distribution'][category] += 1
        
        self._print_distribution_stats(client_data, users_df)
        
        return client_data
    
    def _print_distribution_stats(
        self,
        client_data: Dict,
        users_df: pd.DataFrame
    ):
        """Print statistics about data distribution"""
        print("\n=== Client Data Distribution ===")
        
        # Users per client
        print("\nUsers per client:")
        for client_id in range(self.num_clients):
            num_users = len(client_data[client_id]['users'])
            num_interactions = len(client_data[client_id]['interactions'])
            print(f"  Client {client_id}: {num_users} users, {num_interactions} interactions")
        
        # Preference distribution
        print("\nPreference Type Distribution:")
        pref_matrix = []
        for client_id in range(self.num_clients):
            pref_dist = client_data[client_id]['preference_distribution']
            pref_matrix.append([
                pref_dist.get(pref, 0)
                for pref in users_df['preference_type'].unique()
            ])
        
        pref_df = pd.DataFrame(
            pref_matrix,
            columns=users_df['preference_type'].unique(),
            index=[f"Client {i}" for i in range(self.num_clients)]
        )
        print(pref_df)
        
        # Calculate non-IID metric (standard deviation)
        user_counts = [len(client_data[i]['users']) for i in range(self.num_clients)]
        non_iid_score = np.std(user_counts) / np.mean(user_counts)
        print(f"\nNon-IID Score (CV): {non_iid_score:.3f}")
        print(f"  (Higher = More non-IID)")
    
    def save_client_data(
        self,
        client_data: Dict,
        users_df: pd.DataFrame,
        items_df: pd.DataFrame,
        output_dir: str = './data/simulated_clients'
    ):
        """Save client data to separate directories"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n=== Saving Client Data to {output_path} ===")
        
        for client_id in range(self.num_clients):
            client_dir = output_path / f'client_{client_id}'
            client_dir.mkdir(exist_ok=True)
            
            # Get client users
            client_users = users_df[
                users_df['user_id'].isin(client_data[client_id]['users'])
            ]
            
            # Get client interactions
            client_interactions = client_data[client_id]['interactions']
            
            # Save data
            client_users.to_csv(client_dir / 'users.csv', index=False)
            client_interactions.to_csv(client_dir / 'interactions.csv', index=False)
            
            # Save metadata
            metadata = {
                'client_id': int(client_id),
                'num_users': int(len(client_users)),
                'num_interactions': int(len(client_interactions)),
                'preference_distribution': {
                    k: int(v)
                    for k, v in client_data[client_id]['preference_distribution'].items()
                },
                'category_distribution': {
                    k: int(v)
                    for k, v in client_data[client_id]['category_distribution'].items()
                },
                'alpha': float(self.alpha)
            }       
            
            with open(client_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save global items (shared across all clients)
        items_df.to_csv(output_path / 'items_global.csv', index=False)
        
        print(f"✅ Saved data for {self.num_clients} clients")
    
    def visualize_distribution(
        self,
        client_data: Dict,
        users_df: pd.DataFrame,
        output_dir: str = './data/simulated_clients'
    ):
        """Create visualizations of data distribution"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Users per client
        user_counts = [len(client_data[i]['users']) for i in range(self.num_clients)]
        axes[0, 0].bar(range(self.num_clients), user_counts)
        axes[0, 0].set_title('Users per Client')
        axes[0, 0].set_xlabel('Client ID')
        axes[0, 0].set_ylabel('Number of Users')
        
        # 2. Interactions per client
        interaction_counts = [len(client_data[i]['interactions']) for i in range(self.num_clients)]
        axes[0, 1].bar(range(self.num_clients), interaction_counts)
        axes[0, 1].set_title('Interactions per Client')
        axes[0, 1].set_xlabel('Client ID')
        axes[0, 1].set_ylabel('Number of Interactions')
        
        # 3. Preference distribution heatmap
        pref_matrix = []
        pref_types = users_df['preference_type'].unique()
        for client_id in range(self.num_clients):
            pref_dist = client_data[client_id]['preference_distribution']
            pref_matrix.append([pref_dist.get(pref, 0) for pref in pref_types])
        
        sns.heatmap(
            np.array(pref_matrix).T,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            xticklabels=[f'C{i}' for i in range(self.num_clients)],
            yticklabels=pref_types,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Preference Type Distribution')
        axes[1, 0].set_xlabel('Client ID')
        
        # 4. Category distribution for first 5 clients
        category_matrix = []
        categories = sorted(client_data[0]['category_distribution'].keys())
        for client_id in range(min(5, self.num_clients)):
            cat_dist = client_data[client_id]['category_distribution']
            category_matrix.append([cat_dist.get(cat, 0) for cat in categories])
        
        if category_matrix:
            axes[1, 1].bar(
                range(len(categories)),
                np.array(category_matrix).sum(axis=0),
                tick_label=categories
            )
            axes[1, 1].set_title('Top Categories (First 5 Clients)')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Interactions')
            plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path / 'distribution_visualization.png', dpi=150)
        print(f"✅ Visualization saved to {output_path / 'distribution_visualization.png'}")
        plt.close()


if __name__ == "__main__":
    # Load synthetic data
    data_dir = Path('./data/raw')
    users_df = pd.read_csv(data_dir / 'users.csv')
    items_df = pd.read_csv(data_dir / 'items.csv')
    interactions_df = pd.read_csv(data_dir / 'interactions.csv')
    
    # Parse list columns
    users_df['preferred_categories'] = users_df['preferred_categories'].apply(eval)
    items_df['text_keywords'] = items_df['text_keywords'].apply(eval)
    items_df['image_features'] = items_df['image_features'].apply(eval)
    
    # Create splitter
    splitter = NonIIDDataSplitter(
        num_clients=10,
        alpha=0.5,  # Moderate non-IID
        seed=42
    )
    
    # Split data using Dirichlet
    client_data = splitter.split_by_dirichlet(
        users_df, items_df, interactions_df
    )
    
    # Save client data
    splitter.save_client_data(
        client_data, users_df, items_df
    )
    
    # Visualize distribution
    splitter.visualize_distribution(
        client_data, users_df
    )