"""Routing strategies.

Router interface: select_path(G, s, d, now) -> list[node] | None (None = no path).
The simulator calls it on every retry, so the dynamic router (KAR) sees current key levels.
"""

from __future__ import annotations

import heapq
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


class KARRouter:
    """Key-aware routing: Dijkstra with edge weight 1 / (available_bits(now) + epsilon).

    Links holding more key are cheaper, so traffic is steered around drained links.
    SECOQC's QKD-NL routing used link-state messages carrying each link's key store level
    (Peev 2009 Sec. 4.2), so this has deployed precedent. Levels are read with
    KeyLink.peek(), which changes no state. Called on every retry, so the path follows
    the current key levels (assumes routers see exact, instantaneous state).

    Deterministic tie-breaking: the heap orders equal distances by node, and among
    equal-cost predecessors of a node the smallest one is kept, so the result does not
    depend on edge insertion order.

    Caching (exact, for speed): weights depend only on the time and on the pool states,
    and pool states change only on consume(). Shortest-path trees are therefore cached
    per source for the current (now, total consume count) and reused, e.g. by the many
    requests that retry at the same instant. Any consume or time advance invalidates them.
    """

    def __init__(self, cfg: RoutingConfig | None = None) -> None:
        self.epsilon_bits = (cfg or RoutingConfig()).kar_epsilon_bits
        self._graph: nx.Graph | None = None
        self._adjacency: dict[Hashable, list[tuple[Hashable, Any]]] = {}
        self._links: list[Any] = []
        self._tree_key: tuple[float, int] | None = None
        self._trees: dict[Hashable, dict[Hashable, Hashable]] = {}

    def _prepare(self, G: nx.Graph) -> None:
        if G is not self._graph:
            self._graph = G
            self._adjacency = {
                u: sorted(((v, data["link"]) for v, data in G.adj[u].items()), key=lambda item: item[0])
                for u in G.nodes
            }
            self._links = [link for _, _, link in G.edges(data="link")]
            self._tree_key, self._trees = None, {}

    def select_path(self, G: nx.Graph, s: Hashable, d: Hashable, now: float) -> list[Any] | None:
        if s == d:
            return [s]
        self._prepare(G)
        if s not in self._adjacency or d not in self._adjacency:
            return None
        key = (now, sum(link.consume_count for link in self._links))
        if key != self._tree_key:
            self._tree_key, self._trees = key, {}
        pred = self._trees.get(s)
        if pred is None:
            pred = self._shortest_path_tree(s, now)
            self._trees[s] = pred
        if d not in pred:
            return None
        path = [d]
        while path[-1] != s:
            path.append(pred[path[-1]])
        path.reverse()
        return path

    def _shortest_path_tree(self, s: Hashable, now: float) -> dict[Hashable, Hashable]:
        """Predecessor map of the KAR shortest-path tree from s (s maps to itself)."""
        adjacency = self._adjacency
        eps = self.epsilon_bits
        weight: dict[int, float] = {}  # per link, computed once per tree
        dist: dict[Hashable, float] = {s: 0.0}
        pred: dict[Hashable, Hashable] = {s: s}
        done: set[Hashable] = set()
        heap: list[tuple[float, Hashable]] = [(0.0, s)]
        while heap:
            du, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, link in adjacency[u]:
                if v in done:
                    continue
                w = weight.get(id(link))
                if w is None:
                    w = weight[id(link)] = 1.0 / (link.peek(now) + eps)
                nd = du + w
                old = dist.get(v)
                if old is None or nd < old:
                    dist[v] = nd
                    pred[v] = u
                    heapq.heappush(heap, (nd, v))
                elif nd == old and u < pred[v]:
                    pred[v] = u
        return pred


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
    "kar": KARRouter,
}


def make_router(cfg: RoutingConfig) -> Router:
    try:
        router_cls = ROUTERS[cfg.strategy]
    except KeyError:
        raise ConfigError(
            f"unknown routing.strategy {cfg.strategy!r} (available: {sorted(ROUTERS)})"
        ) from None
    return router_cls(cfg)
