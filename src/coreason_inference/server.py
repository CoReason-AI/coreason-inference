from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from coreason_inference.analysis.active_scientist import ActiveScientist
from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.utils.logger import logger

# Global model store
models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event handler to pre-load models."""
    logger.info("Initializing Service G... Pre-loading models.")

    # Create dummy data to fit a default DynamicsEngine
    # This ensures /simulate/virtual has a model to use if none is provided dynamically
    t = np.linspace(0, 10, 20)
    data = pd.DataFrame({
        "X": np.sin(t),
        "Y": np.cos(t),
        "time": t
    })

    try:
        # Reduced epochs for faster startup since this is just a dummy model
        engine = DynamicsEngine(epochs=1)
        engine.fit(data, time_col="time", variable_cols=["X", "Y"])
        models["default_dynamics"] = engine
        logger.info("Default DynamicsEngine loaded.")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")

    logger.info("Service G Initialized.")
    yield
    models.clear()


app = FastAPI(lifespan=lifespan, title="Service G: Causal Simulation")


# Pydantic Models

class AnalyzeCausalRequest(BaseModel):
    dataset: List[Dict[str, float]]
    variables: List[str]
    method: str = "dynamics"  # "dynamics" or "pc"


class AnalyzeCausalResponse(BaseModel):
    graph: Dict[str, Any]
    metrics: Dict[str, float]


class SimulateVirtualRequest(BaseModel):
    initial_state: Dict[str, float]
    intervention: Optional[Dict[str, float]] = None
    steps: int = 10


class SimulateVirtualResponse(BaseModel):
    trajectory: List[Dict[str, float]]


@app.post("/analyze/causal", response_model=AnalyzeCausalResponse)  # type: ignore
async def analyze_causal(request: AnalyzeCausalRequest) -> AnalyzeCausalResponse:
    """Performs causal discovery on the provided dataset."""
    if not request.dataset:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    df = pd.DataFrame(request.dataset)

    if request.method == "dynamics":
        engine = DynamicsEngine()

        # Ensure time column exists for DynamicsEngine
        time_col = "time"
        if time_col not in df.columns:
            # Use index as time if not provided
            df[time_col] = df.index.astype(float)

        # Ensure variables exist
        for var in request.variables:
            if var not in df.columns:
                raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset")

        try:
            engine.fit(df, time_col=time_col, variable_cols=request.variables)
            graph = engine.discover_loops()

            # Serialize CausalGraph
            graph_dict = graph.model_dump()
            metrics = {"stability_score": graph.stability_score}

            return AnalyzeCausalResponse(graph=graph_dict, metrics=metrics)

        except Exception as e:
            logger.error(f"Dynamics analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e

    elif request.method == "pc":
        scientist = ActiveScientist()

        # Ensure variables exist
        for var in request.variables:
            if var not in df.columns:
                raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset")

        try:
            scientist.fit(df[request.variables])

            # Extract edges from CPDAG
            # causal-learn: -1=TAIL, 1=ARROW, 2=CIRCLE
            # M[i, j] is endpoint at j.
            # i -> j : M[i, j] = 1 (Arrow at j), M[j, i] = -1 (Tail at i)
            matrix = scientist.cpdag
            labels = scientist.labels
            edges = []

            if matrix is not None:
                dim = len(labels)
                for i in range(dim):
                    for j in range(dim):
                        if i == j:
                            continue
                        # Directed Edge i -> j
                        if matrix[i, j] == 1 and matrix[j, i] == -1:
                            edges.append((labels[i], labels[j]))

            graph_dict = {
                "nodes": [{"id": label} for label in labels],
                "edges": edges,
                "loop_dynamics": [],  # PC finds DAGs/CPDAGs (acyclic usually, but CPDAG can represent equivalence)
                "stability_score": 0.0,
            }

            return AnalyzeCausalResponse(graph=graph_dict, metrics={})

        except Exception as e:
            logger.error(f"PC analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e

    else:
        raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")


@app.post("/simulate/virtual", response_model=SimulateVirtualResponse)  # type: ignore
async def simulate_virtual(request: SimulateVirtualRequest) -> SimulateVirtualResponse:
    """Simulates a virtual trajectory given an initial state and intervention."""
    simulator = VirtualSimulator()
    model = models.get("default_dynamics")

    if not model:
        raise HTTPException(status_code=503, detail="Simulation model not initialized.")

    try:
        trajectory = simulator.simulate_trajectory(
            initial_state=request.initial_state,
            steps=request.steps,
            intervention=request.intervention,
            model=model
        )
        return SimulateVirtualResponse(trajectory=trajectory)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
