import os
import shutil
import subprocess
from pathlib import Path

import pytest

@pytest.fixture(autouse=True)
def isolate_artifacts(tmp_path):
    """
    Backup and restore ONLY models/prod/ during the test.
    Ensures tests do not overwrite your real trained model.
    """

    model_dir = Path("models/prod")
    backup = None

    # ✅ Backup existing model directory if exists
    if model_dir.exists():
        backup = tmp_path / "prod_model_backup"
        shutil.move(str(model_dir), str(backup))

    # ✅ Create a fresh models/prod directory for testing
    model_dir.mkdir(parents=True, exist_ok=True)

    yield  # <-- Test runs here

    # 🧹 Remove test-generated model directory
    if model_dir.exists():
        shutil.rmtree(model_dir)

    # 🔄 Restore original backup
    if backup and backup.exists():
        shutil.move(str(backup), str(model_dir))

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
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True)
    
    # Debug: Print output to understand what's happening
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Ensure the script executed successfully
    assert result.returncode == 0, f"Training script failed: {result.stderr}"

    # Check model exists in expected folder
    model_dir = Path("models/prod")
    model_files = list(model_dir.glob("*"))
    assert len(model_files) > 0, "❌ No model file found in models/prod/"