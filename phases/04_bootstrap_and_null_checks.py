"""Run Phase 4 of the report reproduction."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_pipeline import main

if __name__ == "__main__":
    main(["--phases","4",*sys.argv[1:]])
