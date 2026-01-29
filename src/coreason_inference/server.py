from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import pandas as pd
from coreason_identity.models import UserContext
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from coreason_inference.analysis.dynamics import DynamicsEngine
from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.engine import InferenceEngineAsync
from coreason_inference.utils.logger import logger

# Global model store
models: Dict[str, Any] = {}


async def get_user_context(
    x_user_sub: str = Header(..., alias="X-User-Sub"),
    x_user_email: str = Header(..., alias="X-User-Email"),
) -> UserContext:
    """Dependency to extract UserContext from headers (simulated auth)."""
    return UserContext(sub=x_user_sub, email=x_user_email)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan event handler to pre-load models."""
    logger.info("Initializing Service G... Pre-loading models.")

    # Create dummy data to fit a default DynamicsEngine
    # This ensures /simulate/virtual has a model to use if none is provided dynamically
    t = np.linspace(0, 10, 20)
    data = pd.DataFrame({"X": np.sin(t), "Y": np.cos(t), "time": t})

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
async def analyze_causal(
    request: AnalyzeCausalRequest,
    context: UserContext = Depends(get_user_context),  # noqa: B008
) -> AnalyzeCausalResponse:
    """Performs causal discovery on the provided dataset."""
    if not request.dataset:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    df = pd.DataFrame(request.dataset)

    # Ensure time column exists
    time_col = "time"
    if time_col not in df.columns:
        # Use index as time if not provided
        df[time_col] = df.index.astype(float)

    # Ensure variables exist
    for var in request.variables:
        if var not in df.columns:
            raise HTTPException(status_code=400, detail=f"Variable '{var}' not found in dataset")

    try:
        # We delegate to InferenceEngineAsync to enforce identity and unified pipeline
        # and support async execution within FastAPI
        async with InferenceEngineAsync() as engine:
            result = await engine.analyze(data=df, time_col=time_col, variable_cols=request.variables, context=context)

            # Map InferenceResult to AnalyzeCausalResponse
            graph_dict = result.graph.model_dump()
            metrics = {"stability_score": result.graph.stability_score}

            return AnalyzeCausalResponse(graph=graph_dict, metrics=metrics)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", user_id=context.sub)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}") from e


@app.post("/simulate/virtual", response_model=SimulateVirtualResponse)  # type: ignore
async def simulate_virtual(
    request: SimulateVirtualRequest,
    context: UserContext = Depends(get_user_context),  # noqa: B008
) -> SimulateVirtualResponse:
    """Simulates a virtual trajectory given an initial state and intervention."""
    logger.info("Executing virtual simulation", user_id=context.sub)

    simulator = VirtualSimulator()
    model = models.get("default_dynamics")

    if not model:
        raise HTTPException(status_code=503, detail="Simulation model not initialized.")

    try:
        trajectory = simulator.simulate_trajectory(
            initial_state=request.initial_state, steps=request.steps, intervention=request.intervention, model=model
        )
        return SimulateVirtualResponse(trajectory=trajectory)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error(f"Simulation failed: {e}", user_id=context.sub)
        raise HTTPException(status_code=500, detail=str(e)) from e
