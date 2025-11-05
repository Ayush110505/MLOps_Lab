"""
Fix MLflow model stages - Run this if you get the serialization error
"""
import mlflow
from mlflow.tracking import MlflowClient
import time

# Setup
mlflow.set_tracking_uri("file:./mlruns")
client = MlflowClient()

# Get the best model name from your results
# Change this if needed based on which model performed best
best_model = "Ridge_Regression_HousePrices"  # Update this if different

print(f"Fixing model stages for: {best_model}")
print("=" * 60)

try:
    # Get all versions
    model_versions = client.search_model_versions(f"name='{best_model}'")
    latest_version = max([int(mv.version) for mv in model_versions])
    
    print(f"Latest version: {latest_version}")
    print(f"Transitioning to Production...")
    
    # Force transition with string version
    result = client.transition_model_version_stage(
        name=best_model,
        version=str(latest_version),
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"[OK] Successfully transitioned {best_model} v{latest_version} to Production!")
    
except Exception as e:
    print(f"Transition threw error (may still have worked): {e}")
    
    # Verify it worked anyway
    time.sleep(1)
    updated_versions = client.search_model_versions(f"name='{best_model}'")
    prod_versions = [v for v in updated_versions if v.current_stage == "Production"]
    
    if prod_versions:
        print(f"[OK] Verified: Model is in Production stage (version {prod_versions[0].version})")
    else:
        print("[ERROR] Model is NOT in Production stage")
        print("\nTry using MLflow UI:")
        print("1. Run: mlflow ui")
        print("2. Go to http://localhost:5000")
        print("3. Click 'Models' tab")
        print(f"4. Click on '{best_model}'")
        print("5. Click 'Stage' dropdown and select 'Transition to -> Production'")

print("\n" + "=" * 60)
print("Done! Now you can run Section 10 of the notebook.")
