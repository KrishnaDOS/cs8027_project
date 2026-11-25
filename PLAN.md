# PLAN.md — Trusted-Node QKD Network Simulator (Baseline / "Minimum" tier)

## Context
CS 8027 semester project: a network-layer simulation of trusted-node QKD networks.
Only terrestrial networks are in scope; satellite links are out.
Goal for this phase: a working baseline that produces pre-generated comparison plots for a
presentation in ~1 week, built in ~2.5 days of development.
A live demo (notebook or CLI) comes AFTER the plots are done.
No photon-level or quantum-physics simulation: the physical layer is reduced to a
secret-key generation rate for each link.

## Modeling assumptions (copy these into README; reference them in docstrings as A1–A7)
- A1: Each QKD link (graph edge) has a key pool measured in bits. It refills continuously at
  rate R (bits/s) up to a capacity C. Refill is computed lazily whenever the link is accessed:
  pool = min(C, pool + R * (now - last_update)). There are no per-bit events.
- A2: Trusted-node one-time-pad (OTP) relay. Delivering K bits from s to d over path P consumes
  K bits from EVERY link on P.
- A3: Consumption is atomic. A request proceeds only if every link on its path has at least K
  bits at the same instant; then all links are debited together. There is no partial reservation.
- A4: If any link is short of key, the request waits and retries until a timeout. If the
  timeout passes first, the request is recorded as an OUTAGE.
- A5: Waiting requests are not served in FIFO order (a later request may take key first).
  This is a known simplification; document it.
- A6: Latency = waiting time + hops * per_hop_processing_s + path_length_km * propagation_s_per_km.
- A7: Terrestrial networks only.

## Tech stack
Python 3.11+, networkx, simpy, numpy, pandas, matplotlib, pyyaml, pytest.
seaborn is optional; do not require it. Do not use quantum simulators (NetSquid, SeQUeNCe).

## Repository layout
qkd-trusted-node-sim/
  README.md
  requirements.txt
  configs/default.yaml
  data/secoqc.yaml              # SECOQC topology data (real data, see Addendum A.3) (updated per Addendum A)
  src/qkdsim/
    __init__.py
    config.py                   # dataclasses + YAML loading/merging
    links.py                    # KeyLink + rate models
    topologies.py               # chain, secoqc, random_mesh builders
    routing.py                  # SPR, KAR, router registry
    traffic.py                  # Poisson arrivals, pair selection
    simulator.py                # SimPy engine
    metrics.py                  # per-request records -> summaries
    run.py                      # CLI: single run from a config
  experiments/
    configs/e1_chain_hops.yaml
    configs/e2_mesh_load.yaml
    configs/e3_mesh_density.yaml
    run_experiment.py           # expands sweep grid x seeds, runs in parallel
  scripts/make_plots.py
  results/                      # gitignored
  figures/                      # PNG + PDF output
  tests/

## configs/default.yaml (defaults; all values configurable)
simulation: {duration_s: 600, warmup_s: 60, seeds: [0, 1, 2, 3, 4], sample_interval_s: 1.0}
links:
  rate_model: fixed            # fixed | distance | per_link   (fixed is PRIMARY for the presentation) (updated per Addendum A)
  fixed_rate_bps: 5000         # within the 1-10 kbps baseline from the proposal
  distance: {ref_rate_bps: 5000, ref_length_km: 25, attenuation_db_per_km: 0.2}   # (updated per Addendum A)
  #   R(L) = ref_rate_bps * 10^(-alpha * (L - ref_length_km) / 10)
  #   Simplification: real secret-key-rate curves fall faster near maximum range; note in README.
  pool_capacity_bits: 200000
  initial_fill_fraction: 0.5
traffic:
  key_size_bits: 256
  pair_mode: uniform           # uniform | fixed
  # Specify EITHER arrival_rate_per_s OR offered_load (rho). If rho is given:
  #   lambda = rho * sum(R_link) / (K * E[h_spr])
  #   where E[h_spr] is the mean SPR hop count over all ordered node pairs.
  offered_load: 0.5
relay: {timeout_s: 5.0, per_hop_processing_s: 0.001, propagation_s_per_km: 5.0e-6, min_retry_interval_s: 0.001}
routing: {strategy: spr, kar_epsilon_bits: 1.0}   # spr | kar
topology: {type: random_mesh, n_nodes: 30, area_km: 50, target_avg_degree: 4}

## Module specs
links.py
- class KeyLink(rate_bps, capacity_bits, initial_bits, length_km)
  - available(now) -> float
  - consume(now, bits)       # raises if insufficient
  - time_until(now, bits)    # seconds until pool >= bits, ignoring other consumers;
                             # returns inf if bits > capacity
  - counters: total_generated_bits, total_consumed_bits (used by the conservation test)
- link_rate(length_km, cfg) implements the fixed/distance switch.
- KeyLink objects are stored on the nx.Graph as edge attribute "link".

topologies.py (each builder returns an nx.Graph with edge attr "length_km" and optional node attr "pos" in km)
- make_chain(n_nodes=32, spacing_km=63.8)   (updated per Addendum A)
  # Beijing-Shanghai-like line: 32 trusted relay nodes, ~2000 km
- make_secoqc()
  # 6 nodes, 8 links.
  # Loaded read-only from data/secoqc.yaml (real data, see Addendum A.3). (updated per Addendum A)
- make_random_mesh(n_nodes, area_km, target_avg_degree, rng)
  # Random geometric graph; radius chosen to hit the target average degree;
  # edge length = Euclidean distance between node positions.
  # If the graph is disconnected, regenerate (max 100 tries), then fail loudly.
- attach_links(G, cfg, rng) adds a KeyLink to every edge.

routing.py
- Router interface: select_path(G, s, d, now) -> list[node] | None
- SPR: hop-count shortest path; ties broken deterministically (e.g., lexicographic node order).
- KAR: Dijkstra with edge weight = 1 / (link.available(now) + kar_epsilon_bits).
- ROUTERS = {"spr": ..., "kar": ...}

traffic.py
- Poisson arrivals from a numpy Generator.
- Uniform selection of distinct (s, d) pairs; "fixed" mode takes s and d from config.
- No global RNG anywhere: pass np.random.Generator objects, with separate streams for
  topology and for traffic, both derived from the seed.

simulator.py (SimPy)
- generator process: exponential inter-arrival times; spawns handle_request per request.
- handle_request:
    deadline = arrival + timeout_s
    loop:
      path = router.select_path(G, s, d, env.now)      # re-selected on every retry
      if path is None: record outcome="no_path"; return
      if every link on path has available >= K:
          consume K on all links atomically; record outcome="delivered"; return
      wait = max(min_retry_interval_s, max over path links of time_until(env.now, K))
      if env.now + wait > deadline:
          wait until deadline; check once more; if still short, record outcome="outage"; return
      yield env.timeout(wait)
- Record per request: arrival_s, src, dst, outcome, path (final or last attempted), hops,
  path_length_km, wait_s, latency_s (A6), n_attempts.
- Sampler process: every sample_interval_s, record mean pool fill fraction and the fraction
  of links holding fewer than K bits (key-availability time series).
- run_simulation(cfg, seed) -> (records DataFrame, timeseries DataFrame, links snapshot)

metrics.py (computed over post-warmup arrivals only)
- offered requests, delivered count, outage probability, no_path count
- delivered throughput (bits/s) = delivered * K / measured duration
- latency mean / median / p95 (delivered requests only)
- breakdown by hop count: outage probability, mean latency, and count per hop value
- key availability: mean fill fraction; time-averaged fraction of links below K
- report both lambda and the normalized load rho

## Experiments (minimum set)
E1 chain_hops: chain n=32, uniform traffic, SPR; rho in {0.2, 0.5, 0.8}.
   Plots: outage probability vs hop count and mean latency vs hop count, one line per rho.
   NOTE: a single end-to-end flow on a chain is limited only by its bottleneck link, so hop
   count shows little effect. The effect comes from links shared among many flows, hence
   uniform traffic with results broken down by hop count.
   On a chain, SPR and KAR must give identical results (sanity check, also a test).
E2 mesh_load: random mesh n=30, area 50 km, avg degree ~4; SPR vs KAR;
   rho in {0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2}.
   Plots: outage vs rho, throughput vs rho, p95 latency vs rho (one line per router).
E3 mesh_density: fixed rho=0.6; n_nodes in {10, 20, 30, 40, 60} in a fixed 50 km area with a
   fixed connection radius, so average degree grows with n; SPR vs KAR.
   Plots: throughput vs n and outage vs n.
E4 SECOQC: stretch item in M4 (real data available, see Addendum A.3). (updated per Addendum A)
(Later, not this phase: E5 distance rate model.)

Runner: experiments/run_experiment.py <config.yaml>
- Expands the sweep grid x seeds and runs in parallel with
  concurrent.futures.ProcessPoolExecutor(max_workers = cpu_count - 1).
- Writes results/<exp>/summary.csv (one row per run) and results/<exp>/by_hops.csv.
- Raw per-request records are saved only behind a --save-raw flag.

Plots: scripts/make_plots.py <exp> [--all]
- Show the mean across seeds with 95% CI error bars (1.96 * standard error is fine).
- Save PNG (200 dpi) and PDF to figures/.
- Use a consistent color for SPR and for KAR in every figure; label axes with units.
- Provide an option for IEEE single-column width (3.5 in).

## Tests (pytest)
- links: refill math, capacity clamp, consume raises when insufficient, time_until correctness.
- routing: on a small hand-built graph, SPR picks the fewest-hop path and KAR avoids a drained link.
- conservation: after a run, each link's total consumed <= initial + total generated.
- low load: at very low rho, outage ~ 0 and mean latency ~ hops*proc + length*prop (with tolerance).
- chain equivalence: SPR and KAR give identical summaries on a chain with the same seed.
- determinism: the same seed gives an identical summary.

## Milestones (after each: run pytest, stop, summarize, wait for review)
M0 Scaffold: layout, requirements.txt, config dataclasses + YAML loading, pytest runs green.
M1 links.py + topologies.py + their tests.
M2 simulator.py + SPR + traffic + metrics. Smoke test:
   `python -m qkdsim.run --config configs/default.yaml` prints a summary in under ~30 s.
M3 KAR + distance rate model + routing tests + chain-equivalence test.
M4 Experiment runner + E1-E3 configs; full sweeps run in parallel.
M5 make_plots.py produces all E1-E3 figures; README documents how to reproduce,
   lists assumptions A1-A7, and the default parameters.
Out of scope this phase: notebook/CLI demo, "optimal" and "maximum" tier
features (widest-path routing, buffers, multipath, stale state, failures). (updated per Addendum A)

## Conventions
- Type hints everywhere; dataclasses for config.
- Units in variable names (_s, _bps, _bits, _km).
- No global RNG.
- Prefer clarity over abstraction: this is a research baseline, not a framework.
- Performance target: one run (600 s simulated, 30 nodes, rho=1.0) finishes in seconds to tens of seconds.

## Addendum A — Parameter Calibration (from references)
Where this section conflicts with earlier sections, this section wins.
A.1 applies now; the rest applies at the milestone named.

### A.1 Changed defaults (apply to configs/default.yaml now)
| Parameter | Old | New | Source |
|---|---|---|---|
| topology.chain.spacing_km | 65 | 63.8 | Chen 2021 (Nature) Table 1: backbone has 32 trusted relays and 31 links, average link 63.8 km |
| links.distance.ref_length_km | 50 | 25 | Peev 2009 Sec. 3.2: Toshiba link delivered 5.7 kbit/s over 25 km (ref_rate_bps stays 5000) |
| links.rate_model | fixed, distance | fixed, distance, per_link | per_link reads skr_bps from the topology data file (used for SECOQC) |

### A.2 Defaults kept, now with citations (put these in the README)
- key_size_bits = 256: one AES-256 session key per request. Chen 2021 refreshes
  AES-128 keys from QKD key. Peev 2009 Sec. 5.1.1: each AES rekey used 2 x 256 bits.
- fixed_rate_bps = 5000 (metro): Chen 2021 Methods says metro systems typically produce
  ~5 kbps. SECOQC links measured 1-8 kbit/s (see data/secoqc.yaml).
- attenuation_db_per_km = 0.2: Cvitic 2025 Table 1. Deployed links measure ~0.23-0.25 dB/km
  once splices are included (Peev: 7.5 dB over 33 km; Chen: 16.0 dB average over 63.8 km).
  Offer 0.25 as a sensitivity option.
- pool_capacity_bits = 200000: deliberately small. Real key stores were MiB-scale
  (SECOQC: 2-12 MiB, Peev Sec. 5.1.2). Capacity / rate = 40 s of buffered key, so
  depletion happens within a 600 s run. Document this as a modeling choice; it is a
  sensitivity parameter for the optimal tier.
- Chain link rate stays 5 kbps for the minimum tier. Chen 2021 backbone rates averaged
  79.3 kbps (range 28.1-235.4). With rho held fixed, a system k times faster is an exact
  time-compressed copy of the slower one, provided timeout_s is divided by k (capacity in
  bits unchanged, per-hop delays ignored). Outage-vs-hop results therefore carry over and
  waiting time scales by 1/k. An 80 kbps backbone preset is an optimal-tier item.

### A.3 SECOQC topology (experiment E4)
- data/secoqc.yaml is now real data: 6 nodes, 8 links. Use rate_model: per_link.
- When loading, print a warning listing every link whose skr_status is "assumed".
- For rho normalization with per_link rates, sum R over the actual per-link rates.
- E4 is unblocked: SPR vs KAR at rho in {0.3, 0.6, 0.9}, outage and latency bar chart.
  Add it to M4 as a stretch item if time allows.

### A.4 Sources for modeling assumptions (cite in README and docstrings)
- A2 (one-time-pad relay at every hop): ITU-T Y.3800 Sec. 7.1.5 and Sec. 8.2.2 (OTP recommended
  for key relay); Peev 2009 Sec. 2.1 (hop-by-hop mechanism).
- A3/A4 (outage when key runs out): GEANT 2024 Sec. 4.2.2 (an empty key store stops
  transmission on that link).
- KAR has deployed precedent: SECOQC's routing protocol QKD-NL (Peev Sec. 4.2) was OSPF-based,
  with link-state messages carrying each link's average key generation rate and current
  key store amount.
- Bottleneck behaviour: Chen 2021 Fig. 4a reports the key rate between two cities as the
  minimum rate among the intermediate links. Add a test: a single saturating flow on a
  chain delivers throughput close to the minimum link rate.

### A.5 Validation anchors (compare against these; do NOT tune parameters to match)
- Cvitic 2025 (abstract): key-aware routing reduced service rejection by 25-40% compared
  with static shortest path. This range is aggregated across different studies.
  Compare E2 outage at moderate rho and report the result either way.
- Peev 2009 Sec. 5.1.2, Fig. 22: routes switch as key stores hit a minimum threshold.
  Optimal tier: replay this on SECOQC with a single FRM->BRT flow of ~8192 bit/s.
- Peev 2009 Sec. 5.2: max-flow (Ford-Fulkerson) secret capacity. Maximum tier: use as an
  upper bound on throughput.

### A.6 Optimal-tier additions suggested by the references
- Hierarchical metro topology. Chen 2021's metro networks are not random graphs: a few
  trusted relays form a ring or mesh core, and users hang off them as star leaves
  (Beijing: ring of 12 trusted nodes; Jinan: 3 relays serving 50 user nodes). Add
  make_ring_star(n_relays, users_per_relay), with users acting as endpoints only.
  The random geometric mesh stays as the synthetic density baseline.
- reserve_bits (default 0). SECOQC stopped relaying on a link once its store fell to a
  2 MiB minimum kept for authentication (Peev Sec. 5.1.2).
- KAR threshold variant: drop links below a key threshold from the routing graph
  (Cvitic 2025 Sec. 5.1, their ref. [12]).
- Stale routing state: SECOQC's link-state announcements themselves consume key for
  authentication (Peev Sec. 4.2), so update frequency is a real tradeoff.

### A.7 References not yet extracted (none block M0-M5)
Chen 2021 npj (Hefei 46-node network), Sasaki 2011 (Tokyo), Raubitzek 2026 (Austria).

## Addendum B — Mesh and E3 changes (from M1 review)
- Random meshes are generated once and repaired to connectivity by adding the shortest inter-component edges (G.graph["repair_edges"] records how many). This avoids the selection bias of reject-and-redraw, where only ~10% of n=30, degree-4 draws are connected.
- E3 is redesigned: fixed n_nodes=30, 50 km area, target_avg_degree in {3, 4, 6, 8}, SPR vs KAR, rho=0.6. Plots: outage and throughput vs achieved average degree. A node-count sweep at fixed degree is an optional extra.
- KeyLink tracks total_discarded_bits (key lost to a full pool) for later utilization analysis.
