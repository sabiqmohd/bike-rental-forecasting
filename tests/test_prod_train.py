import os
import shutil
import subprocess
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def isolate_artifacts(tmp_path, request):
    """
    Fixture to isolate and clean up model and prediction artifacts.
    """
    model_dir = Path("../models/prod")
    data_dir = Path("../data/prod_data")
    
    # Backup existing directories if they exist
    backups = []
    for path in (model_dir, data_dir):
        if path.exists():
            backup = tmp_path / path.name
            shutil.copytree(str(path), str(backup))
            backups.append((backup, path))
            # Remove original to start fresh
            shutil.rmtree(str(path))
    
    # Create empty directories for the test run
    for path in (model_dir, data_dir):
        path.mkdir(parents=True, exist_ok=True)
    
    yield
    
    # Cleanup: remove test-created directories and restore backups
    for path in (model_dir, data_dir):
        if path.exists():
            shutil.rmtree(str(path))
    
    for backup, original in backups:
        if backup.exists():
            shutil.copytree(str(backup), str(original))

def test_prod_train_creates_model_and_predictions():
    """
    Test Case 1: Run the production training script and verify
    that a model file is saved in /models/prod/ and
    predictions are created in /data/prod_data/.
    """
    # Run the prod_train.py script with explicit PYTHONPATH
    env = os.environ.copy()
    env['PYTHONPATH'] = os.getcwd()  # Add current directory to Python path
    
    result = subprocess.run(
        ["python", "app-ml/entrypoint/train.py"],
        check=False,
        capture_output=True,
        text=True,
        env=env  # Pass the modified environment
    )
    
    # Debug: Print output to understand what's happening
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Ensure the script executed successfully
    assert result.returncode == 0, f"Training script failed: {result.stderr}"

    # Check that a model file exists in models/prod/
    model_dir = Path("../models/prod")
    assert model_dir.exists() and model_dir.is_dir(), "models/prod directory does not exist"
    
    # Look specifically for .cbm files
    model_files = list(model_dir.glob("*.cbm"))
    assert model_files, f"No .cbm model file found in models/prod/. Contents: {list(model_dir.iterdir())}"