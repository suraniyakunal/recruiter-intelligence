import asyncio
import sys
sys.path.insert(0, "backend")  # ensure imports work

from pipelines.ingestion_pipeline import run_ingestion

asyncio.run(run_ingestion("data/raw"))