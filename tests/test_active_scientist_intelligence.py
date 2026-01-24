# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

import numpy as np
from causallearn.graph.Endpoint import Endpoint

from coreason_inference.analysis.active_scientist import ActiveScientist


def test_max_degree_heuristic_star_graph() -> None:
    """
    Test that the Max-Degree Heuristic correctly identifies the 'Hub' in a star graph.

         A
         |
    C -- Hub -- D
         |
         B

    Hub has degree 4 (undirected edges to A, B, C, D).
    Leaves have degree 1.
    Expected: Proposal targets 'Hub'.
    """
    scientist = ActiveScientist()
    labels = ["A", "B", "C", "D", "Hub"]
    n = len(labels)
    hub_idx = 4

    # Create Star Graph (Undirected)
    # M[i, j] = Endpoint at j
    adj = np.zeros((n, n))
    for i in range(4):
        # Undirected edge: Tail at both ends
        adj[i, hub_idx] = Endpoint.TAIL.value
        adj[hub_idx, i] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()

    assert len(proposals) == 1
    proposal = proposals[0]

    # Assert selection
    assert proposal.target == "Hub"
    assert proposal.action == "Intervention_Knockout"
    assert proposal.confidence_gain == "High"

    # Assert Rationale content
    # Should mention degree or heuristic
    assert "degree" in proposal.rationale.lower() or "heuristic" in proposal.rationale.lower()
    assert "4" in proposal.rationale  # The degree count


def test_max_degree_heuristic_chain_graph() -> None:
    """
    Test that the Max-Degree Heuristic identifies the central node in a chain.

    A -- B -- C

    B has degree 2.
    A and C have degree 1.
    Expected: Proposal targets 'B'.
    """
    scientist = ActiveScientist()
    labels = ["A", "B", "C"]

    # Create Chain Graph A-B-C (Undirected)
    # A(0) - B(1)
    # B(1) - C(2)
    adj = np.zeros((3, 3))

    # A-B
    adj[0, 1] = Endpoint.TAIL.value
    adj[1, 0] = Endpoint.TAIL.value

    # B-C
    adj[1, 2] = Endpoint.TAIL.value
    adj[2, 1] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()

    assert len(proposals) == 1
    proposal = proposals[0]

    assert proposal.target == "B"
    assert "2" in proposal.rationale


def test_max_degree_heuristic_fully_oriented() -> None:
    """
    Test that no proposals are generated for a fully oriented graph (DAG).
    """
    scientist = ActiveScientist()
    # A -> B
    adj = np.zeros((2, 2))
    # Endpoint at B is Head, at A is Tail
    # Assuming standard causal-learn/ActiveScientist convention:
    # M[i, j] = endpoint at j
    # Wait, check ActiveScientist._count_oriented_edges:
    # if cpdag[i, j] == ENDPOINT_HEAD and cpdag[j, i] == ENDPOINT_TAIL: Directed i->j
    # So to make A->B: M[0, 1]=HEAD, M[1, 0]=TAIL

    adj[0, 1] = Endpoint.ARROW.value
    adj[1, 0] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = ["A", "B"]

    proposals = scientist.propose_experiments()
    assert len(proposals) == 0


def test_tie_breaking_square_graph() -> None:
    """
    Test tie-breaking in a square graph where all nodes have equal degree.
    A -- B
    |    |
    D -- C

    All have degree 2.
    Expected: Should pick 'A' (index 0) if using argmax/first index.
    """
    scientist = ActiveScientist()
    labels = ["A", "B", "C", "D"]
    # 0, 1, 2, 3
    adj = np.zeros((4, 4))

    # A-B (0-1)
    adj[0, 1] = Endpoint.TAIL.value
    adj[1, 0] = Endpoint.TAIL.value
    # B-C (1-2)
    adj[1, 2] = Endpoint.TAIL.value
    adj[2, 1] = Endpoint.TAIL.value
    # C-D (2-3)
    adj[2, 3] = Endpoint.TAIL.value
    adj[3, 2] = Endpoint.TAIL.value
    # D-A (3-0)
    adj[3, 0] = Endpoint.TAIL.value
    adj[0, 3] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()
    assert len(proposals) == 1
    # Should pick first one in list that has max degree (2).
    # np.argmax returns first occurrence of max.
    # Indices: A=0, B=1, C=2, D=3.
    # Degree: A=2, B=2, C=2, D=2.
    # Argmax should be 0 (A).
    assert proposals[0].target == "A"
    assert "2" in proposals[0].rationale


def test_disconnected_components() -> None:
    """
    Test graph with disconnected components.
    Comp 1: A -- B (Degrees: 1, 1)
    Comp 2: Triangle C--D, D--E, E--C (Degrees: 2, 2, 2)

    Should pick from Component 2 (C, D, or E).
    """
    scientist = ActiveScientist()
    labels = ["A", "B", "C", "D", "E"]
    adj = np.zeros((5, 5))

    # Comp 1: A(0)-B(1)
    adj[0, 1] = Endpoint.TAIL.value
    adj[1, 0] = Endpoint.TAIL.value

    # Comp 2: C(2)-D(3)-E(4)-C(2)
    # C-D
    adj[2, 3] = Endpoint.TAIL.value
    adj[3, 2] = Endpoint.TAIL.value
    # D-E
    adj[3, 4] = Endpoint.TAIL.value
    adj[4, 3] = Endpoint.TAIL.value
    # E-C
    adj[4, 2] = Endpoint.TAIL.value
    adj[2, 4] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()
    assert len(proposals) == 1

    # Expect C (index 2) as it is the first with degree 2.
    assert proposals[0].target == "C"
    assert "2" in proposals[0].rationale


def test_single_undirected_edge() -> None:
    """
    Test minimal case: A -- B.
    Both degree 1. Should pick A.
    """
    scientist = ActiveScientist()
    labels = ["A", "B"]
    adj = np.zeros((2, 2))
    adj[0, 1] = Endpoint.TAIL.value
    adj[1, 0] = Endpoint.TAIL.value

    scientist.cpdag = adj
    scientist.labels = labels

    proposals = scientist.propose_experiments()
    assert len(proposals) == 1
    assert proposals[0].target == "A"
    assert "1" in proposals[0].rationale
