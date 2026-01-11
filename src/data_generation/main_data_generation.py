"""
Main script to run complete data generation pipeline
Usage: python main_data_generation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from synthetic_data_generator import SyntheticDataGenerator
from non_iid_data_splitter import NonIIDDataSplitter
from federated_dataloader import FederatedDataLoader
import pandas as pd
import yaml


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML"""
    
    if config_path is None:
        current_file = Path(__file__).resolve()
        
        project_root = current_file.parent.parent.parent
        
        config_path = project_root / "configs" / "config.yaml"

    print(f"DEBUG: Đang đọc config tại: {config_path}") 
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """Main pipeline"""

    print("=" * 60)
    print("FEDERATED MULTI-MODAL RECOMMENDATION")
    print("Data Generation & Non-IID Distribution Pipeline")
    print("=" * 60)

    # =====================================================
    # LOAD CONFIG
    # =====================================================
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'config.yaml'
    print(f"DEBUG: Đang đọc config tại: {config_path.absolute()}")

    if not config_path.exists():
        print(f"❌ Config file not found at: {config_path}")
        print(f"   Please check the path and try again")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # =====================================================
    # EXTRACT PARAMETERS (SAFE FALLBACKS)
    # =====================================================
    # Data parameters
    num_users = config['data'].get('num_users', 1000)
    num_items = config['data'].get('num_items', 10000)
    num_interactions = config['data'].get('num_interactions', 50000)

    # Federated parameters
    num_clients = config['federated'].get('num_clients', 10)

    # Training parameters
    batch_size = config['training'].get('batch_size', 16)
    test_split = config['training'].get('test_split', 0.2)

    # Non-IID parameters
    non_iid_config = config['data'].get('non_iid', {})
    strategy = non_iid_config.get('strategy', 'dirichlet')
    alpha = non_iid_config.get('alpha', 0.5)

    # Preference distribution
    pref_dist = config['data'].get(
        'preference_distribution',
        {
            'text_heavy': 0.3,
            'image_heavy': 0.3,
            'behavior_heavy': 0.2,
            'balanced': 0.2
        }
    )

    # Output directories (FIX: lấy từ config)
    raw_dir = Path(config['paths'].get('data_raw', './data/raw'))
    output_dir = Path(config['paths'].get('data_dir', './data')) / 'simulated_clients'

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # PRINT CONFIG SUMMARY
    # =====================================================
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Users: {num_users}")
    print(f"Items: {num_items}")
    print(f"Interactions: {num_interactions}")
    print(f"Clients: {num_clients}")
    print(f"Batch size: {batch_size}")
    print(f"Test split: {test_split}")
    print(f"Non-IID strategy: {strategy}")
    print(f"Alpha: {alpha}")
    print(f"Preference distribution: {pref_dist}")

    # =====================================================
    # STEP 1: GENERATE SYNTHETIC DATA
    # =====================================================
    print("\n" + "=" * 60)
    print("STEP 1: Generating Synthetic Dataset")
    print("=" * 60)

    generator = SyntheticDataGenerator(
        num_users=num_users,
        num_items=num_items,
        num_interactions=num_interactions,
        seed=42
    )

    data = generator.generate_all(output_dir=raw_dir)

    users_df = data['users']
    items_df = data['items']
    interactions_df = data['interactions']

    # =====================================================
    # STEP 2: NON-IID SPLITTING
    # =====================================================
    print("\n" + "=" * 60)
    print("STEP 2: Splitting Data with Non-IID Distribution")
    print("=" * 60)

    splitter = NonIIDDataSplitter(
        num_clients=num_clients,
        alpha=alpha,
        seed=42
    )

    if strategy == "dirichlet":
        client_data = splitter.split_by_dirichlet(
            users_df, items_df, interactions_df
        )
    else:
        raise ValueError(f"Unsupported Non-IID strategy: {strategy}")

    splitter.save_client_data(
        client_data,
        users_df,
        items_df,
        output_dir=output_dir
    )

    splitter.visualize_distribution(
        client_data,
        users_df,
        output_dir=output_dir
    )

    # =====================================================
    # STEP 3: TEST FEDERATED DATALOADERS
    # =====================================================
    print("\n" + "=" * 60)
    print("STEP 3: Testing Federated DataLoaders")
    print("=" * 60)

    for client_id in range(min(3, num_clients)):
        print(f"\n--- Testing Client {client_id} ---")

        client_loader = FederatedDataLoader(
            client_id=client_id,
            data_dir=output_dir,
            batch_size=batch_size,
            test_split=test_split,
            seed=42
        )

        train_loader, test_loader = client_loader.create_dataloaders()

        batch = next(iter(train_loader))
        print("✅ Batch loaded successfully")
        print(f"   - User IDs shape: {batch['user_id'].shape}")
        print(f"   - Item IDs shape: {batch['item_id'].shape}")
        print(f"   - Ratings shape: {batch['rating'].shape}")
        print(f"   - Behavior features shape: {batch['behavior_features'].shape}")

    # =====================================================
    # STEP 4: SUMMARY STATISTICS
    # =====================================================
    print("\n" + "=" * 60)
    print("STEP 4: Summary Statistics")
    print("=" * 60)

    print("\n📊 Dataset Summary:")
    print(f"  - Total Users: {len(users_df)}")
    print(f"  - Total Items: {len(items_df)}")
    print(f"  - Total Interactions: {len(interactions_df)}")
    print(
        f"  - Sparsity: "
        f"{1 - len(interactions_df) / (len(users_df) * len(items_df)):.4f}"
    )

    print("\n📊 User Preference Distribution:")
    for pref, count in users_df['preference_type'].value_counts().items():
        print(f"  - {pref}: {count} ({count / len(users_df) * 100:.1f}%)")

    print("\n📊 Client Statistics:")
    for client_id in range(num_clients):
        print(f"  Client {client_id}:")
        print(f"    - Users: {len(client_data[client_id]['users'])}")
        print(f"    - Interactions: {len(client_data[client_id]['interactions'])}")
        print(
            f"    - Preference: "
            f"{dict(client_data[client_id]['preference_distribution'])}"
        )

    print("\n" + "=" * 60)
    print("✅ DATA GENERATION PIPELINE COMPLETED!")
    print("=" * 60)

    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)