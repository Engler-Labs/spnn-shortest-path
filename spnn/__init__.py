"""spnn — stochastic spiking networks for shortest path (Engler 2018, Ch. 4).

A weighted graph is compiled into winner-take-all neuron clusters whose
Boltzmann energy minima coincide with shortest paths; the network is run as a
discrete-time Boltzmann sampler and the path is read out of the induced
subgraph. Per-step cost is O(|E|) via segment reductions, not O(N.d^2).

Public API
----------
Compilation:  ``compile_network``, ``Params``, ``Network``, ``dense_weights``
Simulation:   ``simulate``, ``Trace``, ``energy``, ``membrane``,
              ``sampler_fire``, ``induced_edges``, ``induced_path``
Generators:   ``random_graph``, ``spatial_graph``, ``grid_graph``,
              ``mountain_graph``, ``mountain_spatial_graph``, ``dijkstra``

Typical use, after ``pip install -e .``::

    from spnn import compile_network, Params, simulate, random_graph, dijkstra

    n, edges, w = random_graph(60, 0.05, seed=19)
    net = compile_network(n, edges, w, 0, n - 1, Params())
    trace = simulate(net, steps=1500, seed=0, source=0, target=n - 1,
                     track_paths=True)
"""

from __future__ import annotations

from .compile import Network, Params, compile_network, dense_weights
from .graphs import (
    dijkstra,
    grid_graph,
    mountain_graph,
    mountain_spatial_graph,
    random_graph,
    spatial_graph,
)
from .sim import (
    Trace,
    energy,
    induced_edges,
    induced_path,
    membrane,
    sampler_fire,
    simulate,
)

__all__ = [
    # compilation
    "compile_network",
    "Params",
    "Network",
    "dense_weights",
    # simulation
    "simulate",
    "Trace",
    "energy",
    "membrane",
    "sampler_fire",
    "induced_edges",
    "induced_path",
    # generators
    "random_graph",
    "spatial_graph",
    "grid_graph",
    "mountain_graph",
    "mountain_spatial_graph",
    "dijkstra",
]

__version__ = "0.1.0"
