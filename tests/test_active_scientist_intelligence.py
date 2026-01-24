# Copyright (c) 2026 CoReason, Inc.
# Licensed under the Prosperity Public License 3.0.0

import numpy as np

from coreason_inference.analysis.active_scientist import ENDPOINT_TAIL, ActiveScientist


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
        adj[i, hub_idx] = ENDPOINT_TAIL
        adj[hub_idx, i] = ENDPOINT_TAIL

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
    adj[0, 1] = ENDPOINT_TAIL
    adj[1, 0] = ENDPOINT_TAIL

    # B-C
    adj[1, 2] = ENDPOINT_TAIL
    adj[2, 1] = ENDPOINT_TAIL

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

    from coreason_inference.analysis.active_scientist import ENDPOINT_HEAD

    adj[0, 1] = ENDPOINT_HEAD
    adj[1, 0] = ENDPOINT_TAIL

    scientist.cpdag = adj
    scientist.labels = ["A", "B"]

    proposals = scientist.propose_experiments()
    assert len(proposals) == 0
