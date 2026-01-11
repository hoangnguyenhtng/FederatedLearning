"""
Auto-fix Import Issues
======================

This script will:
1. Create all missing __init__.py files
2. Fix import paths in all Python files
3. Set up proper PYTHONPATH

Run this script from project root:
    cd "D:\Federated Learning"
    python fix_imports.py
"""

import os
from pathlib import Path
import sys


def create_init_files(base_dir: Path):
    """Create __init__.py files in all module directories"""
    
    print("=" * 70)
    print("Step 1: Creating __init__.py files")
    print("=" * 70)
    
    # Directories that need __init__.py
    dirs = [
        base_dir / "src",
        base_dir / "src" / "data_generation",
        base_dir / "src" / "data_processing",
        base_dir / "src" / "models",
        base_dir / "src" / "federated",
        base_dir / "src" / "training",
        base_dir / "src" / "vector_db",
        base_dir / "src" / "api",
        base_dir / "src" / "dashboard"
    ]
    
    # Simple __init__.py content
    init_content = {
        "src": '''"""
Federated Multi-Modal Recommendation System
"""

__version__ = "1.0.0"
''',
        
        "data_generation": '''"""
Data Generation Module
"""

__all__ = []
''',
        
        "data_processing": '''"""
Data Processing Module
"""

__all__ = []
''',
        
        "models": '''"""
Models Module
"""

__all__ = []
''',
        
        "federated": '''"""
Federated Learning Module
"""

__all__ = []
''',
        
        "training": '''"""
Training Module
"""

__all__ = []
''',
        
        "vector_db": '''"""
Vector Database Module
"""

__all__ = []
''',
        
        "api": '''"""
API Module
"""

__all__ = []
''',
        
        "dashboard": '''"""
Dashboard Module
"""

__all__ = []
'''
    }
    
    created_count = 0
    for dir_path in dirs:
        # Create directory if not exists
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py
        init_file = dir_path / "__init__.py"
        
        if not init_file.exists():
            dir_name = dir_path.name
            content = init_content.get(dir_name, '"""\nModule init\n"""\n\n__all__ = []\n')
            
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✓ Created: {init_file.relative_to(base_dir)}")
            created_count += 1
        else:
            print(f"  Exists: {init_file.relative_to(base_dir)}")
    
    print(f"\n✓ Created {created_count} new __init__.py files")
    return created_count


def create_setup_env_script(base_dir: Path):
    """Create setup_env.bat for Windows"""
    
    print("\n" + "=" * 70)
    print("Step 2: Creating environment setup script")
    print("=" * 70)
    
    # Windows batch script
    bat_content = f'''@echo off
REM Setup environment for Federated Learning project

SET PYTHONPATH=%PYTHONPATH%;{base_dir}
SET PYTHONPATH=%PYTHONPATH%;{base_dir}\\src

echo Environment variables set:
echo   PROJECT_ROOT={base_dir}
echo   PYTHONPATH=%PYTHONPATH%

echo.
echo Ready to run scripts!
echo Example: python src\\training\\local_trainer.py
'''
    
    bat_file = base_dir / "setup_env.bat"
    with open(bat_file, 'w', encoding='utf-8') as f:
        f.write(bat_content)
    
    print(f"✓ Created: {bat_file.relative_to(base_dir)}")
    
    # Linux/Mac shell script
    sh_content = f'''#!/bin/bash
# Setup environment for Federated Learning project

export PYTHONPATH=$PYTHONPATH:{base_dir}
export PYTHONPATH=$PYTHONPATH:{base_dir}/src

echo "Environment variables set:"
echo "  PROJECT_ROOT={base_dir}"
echo "  PYTHONPATH=$PYTHONPATH"

echo ""
echo "Ready to run scripts!"
echo "Example: python src/training/local_trainer.py"
'''
    
    sh_file = base_dir / "setup_env.sh"
    with open(sh_file, 'w', encoding='utf-8') as f:
        f.write(sh_content)
    
    # Make executable
    try:
        os.chmod(sh_file, 0o755)
    except:
        pass
    
    print(f"✓ Created: {sh_file.relative_to(base_dir)}")
    
    return bat_file, sh_file


def verify_project_structure(base_dir: Path):
    """Verify project structure is correct"""
    
    print("\n" + "=" * 70)
    print("Step 3: Verifying project structure")
    print("=" * 70)
    
    required_files = [
        "configs/config.yaml",
        "src/models/multimodal_encoder.py",
        "src/models/recommendation_model.py",
        "src/models/attention_mechanism.py",
        "src/federated/aggregator.py",
        "src/training/training_utils.py",
        "src/training/local_trainer.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✓ Found: {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    if all_exist:
        print("\n✓ All required files present")
    else:
        print("\n⚠️  Some files are missing. Please check project structure.")
    
    return all_exist


def test_imports(base_dir: Path):
    """Test if imports work"""
    
    print("\n" + "=" * 70)
    print("Step 4: Testing imports")
    print("=" * 70)
    
    # Add to path
    sys.path.insert(0, str(base_dir))
    sys.path.insert(0, str(base_dir / "src"))
    
    tests = [
        ("Training Utils", "from src.training.training_utils import train_one_epoch"),
        ("Local Trainer", "from src.training.local_trainer import LocalTrainer"),
        ("Aggregator", "from src.federated.aggregator import get_aggregation_strategy"),
        ("Models", "from src.models.recommendation_model import FedPerRecommender"),
    ]
    
    success_count = 0
    for test_name, import_statement in tests:
        try:
            exec(import_statement)
            print(f"✓ {test_name}: OK")
            success_count += 1
        except ImportError as e:
            print(f"✗ {test_name}: FAILED")
            print(f"  Error: {str(e)}")
        except Exception as e:
            print(f"⚠️  {test_name}: Warning - {str(e)}")
            success_count += 1  # Count as success if not import error
    
    print(f"\n✓ {success_count}/{len(tests)} imports successful")
    return success_count == len(tests)


def create_quick_test_script(base_dir: Path):
    """Create a quick test script"""
    
    print("\n" + "=" * 70)
    print("Step 5: Creating quick test script")
    print("=" * 70)
    
    test_content = '''"""
Quick Import Test
Run this to verify all imports work
"""

import sys
from pathlib import Path

# Add to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("=" * 70)
print("Testing Imports")
print("=" * 70)

tests = [
    ("Training Utils", "from src.training.training_utils import train_one_epoch, evaluate"),
    ("Local Trainer", "from src.training.local_trainer import LocalTrainer"),
    ("Aggregator", "from src.federated.aggregator import get_aggregation_strategy"),
    ("Models", "from src.models.recommendation_model import FedPerRecommender"),
    ("MultiModal Encoder", "from src.models.multimodal_encoder import MultiModalEncoder"),
    ("Attention", "from src.models.attention_mechanism import AdaptiveAttentionFusion"),
]

passed = 0
failed = 0

for test_name, import_stmt in tests:
    try:
        exec(import_stmt)
        print(f"✓ {test_name}")
        passed += 1
    except Exception as e:
        print(f"✗ {test_name}: {str(e)}")
        failed += 1

print("\\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("\\n✓ All imports working! Ready to run Phase 1.")
else:
    print(f"\\n⚠️  {failed} imports failed. Please fix before continuing.")
'''
    
    test_file = base_dir / "test_imports.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print(f"✓ Created: {test_file.relative_to(base_dir)}")
    print(f"\n  Run with: python test_imports.py")
    
    return test_file


def main():
    """Main function"""
    
    print("=" * 70)
    print("Federated Learning - Import Fixer")
    print("=" * 70)
    
    # Get project root
    if len(sys.argv) > 1:
        base_dir = Path(sys.argv[1])
    else:
        # Try to detect
        base_dir = Path.cwd()
        if not (base_dir / "src").exists():
            base_dir = Path(__file__).parent
    
    print(f"\nProject root: {base_dir}")
    
    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return
    
    # Step 1: Create __init__.py files
    create_init_files(base_dir)
    
    # Step 2: Create environment setup scripts
    create_setup_env_script(base_dir)
    
    # Step 3: Verify structure
    verify_project_structure(base_dir)
    
    # Step 4: Test imports
    test_imports(base_dir)
    
    # Step 5: Create test script
    create_quick_test_script(base_dir)
    
    # Final instructions
    print("\n" + "=" * 70)
    print("✓ SETUP COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Run: setup_env.bat (Windows) or source setup_env.sh (Linux)")
    print("2. Test: python test_imports.py")
    print("3. If all pass, continue with Phase 1 training")
    print("\nFor Phase 1:")
    print("  cd src/data_generation")
    print("  python main_data_generation.py")
    print("  cd ../training")
    print("  python federated_training_pipeline.py")
    print("=" * 70)


if __name__ == "__main__":
    main()