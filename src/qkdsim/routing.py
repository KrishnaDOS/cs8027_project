"""Routing strategies.

Router interface: select_path(G, s, d, now) -> list[node] | None (None = no path).
The simulator calls it on every retry, so dynamic routers (KAR, M3) see current key levels.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Hashable, Protocol

import networkx as nx

from qkdsim.config import ConfigError, RoutingConfig


class Router(Protocol):
    def select_path(self, G: nx.Graph, s: Hashable, d: Hashable, now: float) -> list[Any] | None: ...


class SPRRouter:
    """Static shortest-path routing by hop count.

    Ties are broken deterministically: among all minimum-hop paths, the one whose node
    sequence is lexicographically smallest (nodes compared in their natural sort order).
    Paths ignore key levels, so they are cached; the topology must not change during a run.
    """

    def __init__(self, cfg: RoutingConfig | None = None) -> None:
        self._graph: nx.Graph | None = None
        self._dist_to: dict[Hashable, dict[Hashable, int]] = {}
        self._paths: dict[tuple[Hashable, Hashable], list[Any] | None] = {}

    def select_path(self, G: nx.Graph, s: Hashable, d: Hashable, now: float) -> list[Any] | None:
        if G is not self._graph:  # new graph: drop caches
            self._graph, self._dist_to, self._paths = G, {}, {}
        key = (s, d)
        if key not in self._paths:
            self._paths[key] = self._compute(G, s, d)
        return self._paths[key]

    def _compute(self, G: nx.Graph, s: Hashable, d: Hashable) -> list[Any] | None:
        if s == d:
            return [s]
        dist = self._dist_to.get(d)
        if dist is None:
            dist = _bfs_hops(G, d)
            self._dist_to[d] = dist
        if s not in dist:
            return None
        # Walk from s towards d, always taking the smallest neighbour that is one hop closer.
        path = [s]
        node = s
        while node != d:
            node = min(nbr for nbr in G.neighbors(node) if dist.get(nbr) == dist[node] - 1)
            path.append(node)
        return path


def _bfs_hops(G: nx.Graph, root: Hashable) -> dict[Hashable, int]:
    dist = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for nbr in G.neighbors(node):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


ROUTERS: dict[str, type] = {
    "spr": SPRRouter,
    # "kar": KARRouter,  # Milestone 3
}


def make_router(cfg: RoutingConfig) -> Router:
    try:
        router_cls = ROUTERS[cfg.strategy]
    except KeyError:
        raise ConfigError(
            f"routing.strategy {cfg.strategy!r} is not implemented yet "
            f"(available: {sorted(ROUTERS)})"
        ) from None
    return router_cls(cfg)
