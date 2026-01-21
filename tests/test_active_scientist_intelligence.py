import numpy as np
import pandas as pd
import pytest
from coreason_inference.analysis.active_scientist import ActiveScientist


@pytest.fixture
def chain_data() -> pd.DataFrame:
    """
    Generates data for A - B - C chain.
    Markov Equivalent to A->B->C, A<-B<-C, A<-B->C.
    The skeleton is A-B, B-C.
    All edges are undirected in the CPDAG.
    """
    np.random.seed(42)
    n = 1000
    a = np.random.randn(n)
    b = 0.5 * a + np.random.randn(n) * 0.1
    c = 0.5 * b + np.random.randn(n) * 0.1

    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_select_maximal_intervention(chain_data: pd.DataFrame) -> None:
    """
    Test that ActiveScientist selects the intervention that resolves the most ambiguity.
    In A - B - C:
    - Node A has 1 undirected edge (A-B)
    - Node B has 2 undirected edges (A-B, B-C)
    - Node C has 1 undirected edge (B-C)

    The heuristic should prefer B.
    """
    scientist = ActiveScientist()
    scientist.fit(chain_data)

    proposals = scientist.propose_experiments()

    # Expect only ONE proposal (The best one)
    assert len(proposals) == 1

    proposal = proposals[0]
    # Expect target to be B
    assert proposal.target == "B"
    assert proposal.confidence_gain == "High"
    # Case insensitive check or robust string match
    assert "resolve 2" in proposal.rationale.lower()
    assert "ambiguous edges" in proposal.rationale.lower()


def test_no_ambiguity_returns_empty_list() -> None:
    """
    Test that if the graph is fully resolved (e.g., V-structure A -> B <- C),
    no experiments are proposed.
    """
    np.random.seed(42)
    n = 1000
    a = np.random.randn(n)
    c = np.random.randn(n)
    # B depends on A and C (Collider)
    b = a + c + np.random.randn(n) * 0.1

    df = pd.DataFrame({"A": a, "B": b, "C": c})

    scientist = ActiveScientist()
    scientist.fit(df)

    proposals = scientist.propose_experiments()
    assert proposals == []
