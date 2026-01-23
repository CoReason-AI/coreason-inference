from unittest.mock import MagicMock

import pandas as pd

from coreason_inference.analysis.dynamics import ODEFunc
from coreason_inference.analysis.virtual_simulator import VirtualSimulator
from coreason_inference.schema import CausalGraph, ProtocolRule


def test_dynamics_linear_attribute_removed() -> None:
    """
    Verify that ODEFunc no longer has a 'linear' attribute, which was replaced by 'net'.
    This test confirms the refactoring was complete and no lingering references exist.
    """
    model = ODEFunc(input_dim=5, hidden_dim=10)
    assert hasattr(model, "net")
    assert hasattr(model, "W")
    # Should not have 'linear'
    assert not hasattr(model, "linear")


def test_virtual_simulator_operator_map() -> None:
    """
    Test that VirtualSimulator uses the OPERATOR_MAP efficiently.
    This replaces a potential huge if/else block.
    """
    sim = VirtualSimulator()
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

    # Test '>'
    rule = ProtocolRule(feature="val", operator=">", value=3, rationale="test")
    res = sim._apply_rules(df, [rule])
    assert len(res) == 2
    assert res["val"].tolist() == [4, 5]

    # Test invalid operator handling (graceful failure)
    rule_bad = ProtocolRule(feature="val", operator="??", value=3, rationale="test")
    res_bad = sim._apply_rules(df, [rule_bad])
    assert len(res_bad) == 5  # Should ignore bad rule


def test_inference_engine_dependency_injection() -> None:
    """
    Verify we can inject custom components into InferenceEngine.
    This improves testability and decoupling.
    """
    from coreason_inference.engine import InferenceEngine

    mock_dynamics = MagicMock()
    mock_active_scientist = MagicMock()
    mock_latent_miner = MagicMock()

    # Mock latent miner to return a small DF
    mock_latent_miner.discover_latents.return_value = pd.DataFrame({"Z": [0, 0]})

    # Mock dynamics to return a valid CausalGraph (Pydantic validation requires this)
    dummy_graph = CausalGraph(nodes=[], edges=[], loop_dynamics=[], stability_score=0.0)
    mock_dynamics.discover_loops.return_value = dummy_graph

    engine = InferenceEngine(
        dynamics_engine=mock_dynamics, active_scientist=mock_active_scientist, latent_miner=mock_latent_miner
    )

    assert engine.dynamics_engine == mock_dynamics

    # Test analyze call propagates to mock
    df = pd.DataFrame({"t": [0, 1], "A": [1, 2]})
    engine.analyze(df, "t", ["A"])

    mock_dynamics.fit.assert_called_once()
    mock_dynamics.discover_loops.assert_called_once()

    mock_active_scientist.fit.assert_called_once()
