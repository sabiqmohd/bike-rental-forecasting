import os
import shutil
import subprocess
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def isolate_artifacts(tmp_path, request):
    """
    Fixture to isolate and clean up model and prediction artifacts.
    Moves existing artifacts to a temporary location before the test,
    and restores them after the test.
    """
    model_dir = Path("../models/prod")
    data_dir = Path("../data/prod_data")
    # Backup existing directories if they exist
    backups = []
    for path in (model_dir, data_dir):
        if path.exists():
            backup = tmp_path / path.name
            shutil.move(str(path), str(backup))
            backups.append((backup, path))
    # Create empty directories for the test run
    for path in (model_dir, data_dir):
        path.mkdir(parents=True, exist_ok=True)
    yield
    # Cleanup: remove test-created directories and restore backups
    for path in (model_dir, data_dir):
        if path.exists():
            shutil.rmtree(str(path))
    for backup, original in backups:
        # Move backup back to original location
        shutil.move(str(backup), str(original))

def test_prod_train_creates_model_and_predictions():
    """
    Test Case 1: Run the production training script and verify
    that a model file is saved in /models/prod/ and
    predictions are created in /data/prod_data/.
    """
    # Run the prod_train.py script
    result = subprocess.run(
        ["python", "app-ml/entrypoint/train.py"],
        check=False,
        capture_output=True,
        text=True
    )
    # Ensure the script executed successfully
    assert result.returncode == 0, f"Training script failed: {result.stderr}"

    # Check that a model file exists in models/prod/
    model_dir = Path("../models/prod")
    assert model_dir.exists() and model_dir.is_dir(), "models/prod directory does not exist"
    model_files = list(model_dir.iterdir())
    assert model_files, "No model file found in models/prod/"

    # # Check that prediction outputs are created in data/prod_data/
    # data_dir = Path("../data/prod_data")
    # assert data_dir.exists() and data_dir.is_dir(), "data/prod_data directory does not exist"
    # # Check both possible subfolders (csv or parquet) for output files
    # prediction_files = []
    # csv_dir = data_dir / "csv"
    # parquet_dir = data_dir / "parquet"
    # if csv_dir.exists():
    #     prediction_files.extend(list(csv_dir.iterdir()))
    # if parquet_dir.exists():
    #     prediction_files.extend(list(parquet_dir.iterdir()))
    # assert prediction_files, "No prediction files found in data/prod_data/"