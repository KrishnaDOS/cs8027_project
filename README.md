# Trusted-node QKD network simulator

KSU CS 8027 (Advanced Networking & Architecture), Fall 2026, final project.

A network-layer, discrete-event simulator (SimPy) of terrestrial trusted-node quantum key
distribution (QKD) networks. Each QKD link is reduced to a secret-key pool that refills at
a key rate; requests for a session key are relayed hop by hop with one-time-pad encryption
and fail (outage) if the key along their path does not become available before a timeout.
The simulator compares static shortest-path routing (SPR) with key-aware routing (KAR) on a
chain, random meshes and the SECOQC Vienna network. There is no photon-level physics: the
physical layer is only a key rate per link.

## Main results

Numbers are means across seeds with 95% t-intervals, taken from
[figures/tables.md](figures/tables.md).

- **Routing only helps where paths are diverse (E2).** On the default 30-node mesh
  (`mesh0`, topology seed 0), SPR's busiest link is a *bridge* whose removal would cut off
  43.4% of node pairs, so no router can avoid it: SPR and KAR give the same outage at every
  load except ρ = 1 (0.523 vs 0.519). On the `diverse` mesh (seed 8, largest bridge share
  0.129), KAR removes outage entirely for 0.2 ≤ ρ ≤ 0.4 (SPR: 0.039–0.220) and roughly
  halves it at ρ = 0.6 (0.315 → 0.149).
- **KAR's gain grows with mesh density (E3, ρ = 0.3, 20 topologies per degree).** The
  relative outage reduction (1 − mean KAR / mean SPR) rises from 8.1% at achieved degree
  3.31 to 18.0% (4.21), 45.6% (6.04) and 73.4% (8.11), while the mean largest bridge share
  falls from 0.465 to 0.043. SPR's own outage is not monotonic in degree (0.143, 0.212,
  0.215, 0.164): ρ normalizes by total link rate, so at fixed ρ a denser mesh also carries
  more traffic and is relatively more loaded at its bottleneck (mean `rho_sat` falls from
  0.216 to 0.129).
- **At a fixed arrival rate (E3b, λ = 95.1826 req/s for every degree)**, more links do
  reduce outage: SPR 0.320 → 0.229 → 0.086 → 0.015 and KAR 0.306 → 0.188 → 0.038 → 0.003
  from degree 3.31 to 8.11, with relative reductions of 4.4%, 17.8%, 55.6% and 81.8%. At
  degree 8 the 81.8% is a ratio of two small numbers: absolute outage is 0.015 ± 0.025 for
  SPR and 0.003 ± 0.004 for KAR, a difference of about 1.2 percentage points, and SPR's
  interval includes zero.
- **Headline figure** (`e3_gain_vs_bridge_share`): each topology's KAR gain against its
  largest bridge share, for E3 and E3b. Large gains occur only on topologies whose largest
  bridge carries a small share of node pairs; meshes dominated by one bridge gain little.
  Topologies where SPR has no outage are omitted (the reduction is undefined there); the
  panel titles give the counts.
- **SPR concentrates load.** Links saturate far below ρ = 1: SPR's busiest link saturates
  at ρ_sat = 0.688 on the 32-node chain, 0.100 on SECOQC, and on average 0.171 on the
  degree-4 E3 meshes (0.216 at degree 3 down to 0.129 at degree 8).
- **Chain (E1).** Below ρ_sat (ρ = 0.65) there is no outage; above it, outage is 0.057 at
  ρ = 0.8 and 0.149 at ρ = 0.95 and grows with hop count (`e1_outage_vs_hops`). On a chain
  SPR = KAR; the single KAR run (ρ = 0.8, seed 0) is identical to SPR on the same seed.
- **SECOQC (E4).** No outage at ρ = 0.05 and 0.1; 0.063 at ρ = 0.15 and 0.216 at ρ = 0.3,
  identical for SPR and KAR at every load. In a diagnostic run (ρ = 0.3, seed 0, both
  routers) every outage involved St. Poelten (STP), whose only link, STP–BRT (700 bit/s
  over 85 km, a bridge), held less than one key (256 bits) 98.8% of the time.

**Confidence intervals in E2** reflect traffic randomness on one fixed topology per mesh
only (the 5 seeds vary arrivals and node pairs, not the graph), so they are narrow and say
nothing about other topologies; E3/E3b vary the topology and are the guide to that spread.

### E3 with 20 vs 5 topology seeds

E3 was first run with 5 topologies per degree, then re-run with 20 (Addendum E). The
qualitative result held, and the estimates tightened:

| Target degree | Relative reduction, 5 → 20 seeds | SPR outage, 5 → 20 seeds | KAR outage, 5 → 20 seeds | Achieved degree, 5 → 20 |
|---|---|---|---|---|
| 3 | 5.6% → 8.1% | 0.129 ± 0.080 → 0.143 ± 0.044 | 0.122 ± 0.085 → 0.131 ± 0.047 | 3.24 → 3.31 |
| 4 | 17.3% → 18.0% | 0.197 ± 0.124 → 0.212 ± 0.053 | 0.163 ± 0.162 → 0.174 ± 0.062 | 4.20 → 4.21 |
| 6 | 40.1% → 45.6% | 0.178 ± 0.093 → 0.215 ± 0.053 | 0.106 ± 0.132 → 0.117 ± 0.053 | 6.27 → 6.04 |
| 8 | 74.9% → 73.4% | 0.143 ± 0.044 → 0.164 ± 0.039 | 0.036 ± 0.100 → 0.044 ± 0.030 | 8.69 → 8.11 |

Intervals shrank by roughly half (KAR at degree 8: ± 0.100 → ± 0.030), and SPR's peak at
degrees 4–6 persisted. The first 5 seeds happened to give denser-than-average meshes at
target degrees 6 and 8 (achieved 6.27 and 8.69 vs 6.04 and 8.11 over 20 seeds). The
5-seed values come from `figures/tables.md` at commit `9e549eb`.

### Provenance

E2 results were produced at commit `5750866`; E1, E3, E3b and E4 at `3547f95`, all from
clean working trees (`meta.json`). `git diff 5750866 3547f95` shows **no change** to the
simulation code (`simulator.py`, `links.py`, `routing.py`, `traffic.py`, `topologies.py`,
`metrics.py`), nor to `configs/` or `data/`; only `analysis.py` (plots and tables) and
`experiment.py` (an aggregation warning) changed. The two commits therefore simulate the
same model.

## Modeling assumptions

| | Assumption | Source |
|---|---|---|
| A1 | Each link has a key pool (bits) refilled continuously at rate R up to capacity C, computed lazily: `pool = min(C, pool + R·Δt)`. No per-bit events. | modeling choice |
| A2 | Trusted-node one-time-pad relay: delivering K bits over path P consumes K bits on **every** link of P. | ITU-T Y.3800 Sec. 7.1.5 and 8.2.2 (OTP recommended for key relay); Peev et al. 2009 Sec. 2.1 (hop-by-hop mechanism) |
| A3 | Consumption is atomic: a request proceeds only when every link on its path holds ≥ K bits at the same instant; no partial reservation. | GÉANT 2024 Sec. 4.2.2 (an empty key store stops transmission on that link) |
| A4 | Otherwise the request waits and retries until `timeout_s`; then it is an **outage**. | GÉANT 2024 Sec. 4.2.2 |
| A5 | Waiting requests are not served FIFO (a later request may take key first). | known simplification |
| A6 | Latency = waiting time + hops × `per_hop_processing_s` + path length × `propagation_s_per_km`. | modeling choice |
| A7 | Terrestrial networks only (no satellite links). | project scope |

KAR has deployed precedent: SECOQC's routing protocol QKD-NL was OSPF-based, with link-state
messages carrying each link's key generation rate and current key store level (Peev et al.
2009 Sec. 4.2). Here KAR runs Dijkstra with edge weight `1 / (available_bits + ε)` and sees
exact, instantaneous key levels; SPR is hop-count shortest path with deterministic
tie-breaking. A single end-to-end flow is limited by its slowest link, as Chen et al. 2021
(Fig. 4a) report for city-to-city key rates; a test checks this.

## Default parameters

Defaults live in [configs/default.yaml](configs/default.yaml) (mirrored by the dataclasses in
`src/qkdsim/config.py`).

| Parameter | Default | Rationale / source |
|---|---|---|
| `traffic.key_size_bits` | 256 | One AES-256 session key per request. Chen 2021 refreshes AES-128 keys from QKD key; Peev 2009 Sec. 5.1.1 used 2 × 256 bits per AES rekey. |
| `links.fixed_rate_bps` | 5000 | Metro QKD systems typically produce ~5 kbit/s (Chen 2021, Methods); SECOQC links measured 1–8 kbit/s (`data/secoqc.yaml`). |
| `links.rate_model` | `fixed` | Alternatives: `distance`, R(L) = 5000 · 10^(−α(L − 25 km)/10), anchored at Peev 2009 Sec. 3.2 (5.7 kbit/s over 25 km); `per_link` (measured rates from the topology file, used for SECOQC). Real secret-key-rate curves fall faster near maximum range than this exponential. |
| `links.distance.attenuation_db_per_km` | 0.2 | Cvitic 2025 Table 1. Deployed links measure ~0.23–0.25 dB/km with splices (Peev: 7.5 dB over 33 km; Chen: 16.0 dB average over 63.8 km); 0.25 is a sensitivity option. |
| `links.pool_capacity_bits` | 200 000 | Deliberately small (40 s of key at 5 kbit/s) so depletion happens within a 600 s run. Real key stores were MiB-scale (SECOQC 2–12 MiB, Peev Sec. 5.1.2). A modeling choice and sensitivity parameter. |
| `links.initial_fill_fraction` | 0.5 | Pools start half full. |
| `topology.chain.spacing_km` | 63.8 | Beijing–Shanghai backbone: 32 trusted relays, 31 links, average 63.8 km (Chen 2021, Table 1). Chain link rate stays 5 kbit/s (Chen's backbone averaged 79.3 kbit/s); at fixed ρ a k-times faster system is a time-compressed copy if `timeout_s` is divided by k. |
| `topology` (mesh) | 30 nodes, 50 km square, target degree 4 | Random geometric graph; radius from the exact boundary-corrected pair probability; disconnected draws repaired with the shortest inter-component edges (Addendum B). |
| `relay.timeout_s` | 5.0 | Outage deadline. |
| `relay.per_hop_processing_s` / `propagation_s_per_km` | 0.001 / 5e-6 | A6. |
| `simulation` | 600 s, 60 s warm-up, seeds 0–4 | Metrics use arrivals in [warm-up, duration). |
| `traffic.offered_load` (ρ) | 0.5 | λ = ρ · ΣR / (K · E[h_SPR]): ρ = 1 means total key generation equals the key consumed on shortest paths. In fixed-pair mode ρ is normalized by the pair's bottleneck rate instead. |

Sources: Peev et al., *The SECOQC quantum key distribution network in Vienna*, New J. Phys.
11, 075001 (2009); Chen et al., *An integrated space-to-ground quantum communication network
over 4,600 kilometres*, Nature 589, 214 (2021); ITU-T Y.3800; GÉANT (2024); Cvitic (2025).
The last three are cited by the short names used in PLAN.md (Addendum A). The SECOQC topology and per-link rates in `data/secoqc.yaml` come from Peev 2009
(each link records its source; two rates are *assumed* and a warning lists them at load time).

## Install

Python 3.11 or newer (developed and tested on CPython 3.15.2, Windows 11).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Test

```bash
python -m pytest            # 136 tests, under a minute
```

## Run

One simulation:

```bash
python -m qkdsim.run --config configs/default.yaml
python -m qkdsim.run --config configs/default.yaml --set routing.strategy=kar --set traffic.offered_load=0.3 --by-hops
```

`--set` takes dotted keys (values parsed as YAML). The summary includes λ and ρ, the SPR
bottleneck diagnostics (`rho_bn`, `rho_sat`, bridges), outage, throughput, latency and key
utilization.

## Experiments

| Config | What it varies | Runs (full / quick) |
|---|---|---|
| `e1_chain_hops.yaml` | chain (32 nodes), ρ ∈ {0.65, 0.8, 0.95} around ρ_sat = 0.688, SPR; one KAR sanity run at ρ = 0.8 | 16 / 10 |
| `e2_mesh_load.yaml` | two fixed meshes × ρ ∈ {0.05 … 1.0} (9 values) × SPR/KAR | 180 / 108 |
| `e3_mesh_density.yaml` | target degree ∈ {3, 4, 6, 8}, ρ = 0.3, 20 topologies, SPR/KAR | 160 / 24 |
| `e3b_mesh_density_fixed_lambda.yaml` | as E3, but λ = 95.1826 req/s fixed (E3's mean degree-4 λ) | 160 / 24 |
| `e4_secoqc.yaml` | SECOQC with measured rates, ρ ∈ {0.05, 0.1, 0.15, 0.3} around ρ_sat = 0.100, SPR/KAR | 40 / 24 |

Use `--dry-run` to time the most expensive run and project the sweep's wall-clock.

```bash
python experiments/run_experiment.py experiments/configs/e2_mesh_load.yaml --dry-run   # time + projection only
python experiments/run_experiment.py experiments/configs/e2_mesh_load.yaml --quick     # 300 s, seeds 0-2
python experiments/run_experiment.py experiments/configs/e2_mesh_load.yaml             # full sweep
```

Options: `--workers N` (default: CPU count − 1), `--results-dir DIR`. Each run is written to
`results/<exp>/runs/<run_id>.json` as soon as it finishes; rerunning skips completed runs
(resume), keyed by a hash of the run's full configuration and seed. **Code changes do not
invalidate results**: delete the results folder after changing simulator code. Aggregates
go to `summary.csv` (one row per run) and `by_hops.csv`; `meta.json` records the config
hash, git commit (and whether the tree was dirty), start/end time and Python version.
`--quick` writes to `results/<exp>_quick/`. Aggregation includes only run files whose
run id belongs to the current config's grid; others (e.g. from an earlier grid) are
reported with a warning, skipped, and left on disk.

SPR and KAR always run on identical seeds, so results can be paired. E2 fixes each mesh with
`topology.seed` and varies only traffic seeds; E3 and E3b leave `topology.seed` unset, so the
run seed also picks the topology, and seed s gives the same mesh in E3 and E3b.

**E3 vs E3b.** ρ is normalized by total link rate, and denser meshes add mostly peripheral
capacity, so at a fixed ρ they carry more traffic and are relatively more loaded at their
bottleneck. E3b holds the arrival rate fixed instead, at λ = 95.1826 req/s, the mean λ of the
E3 degree-4 runs at ρ = 0.3 (derivation in the config file), so sparser meshes see more load
than in E3 and denser meshes less.

## Figures and tables

```bash
python scripts/make_plots.py --all            # PNG (200 dpi) + PDF in figures/
python scripts/make_plots.py --all --ieee     # 3.5 in single-column versions, *_ieee.*
python scripts/make_plots.py --exp e2         # one experiment (e1, e2, e3, e3b, e4, topo)
python scripts/make_tables.py                 # figures/tables.md
```

Both read `results/<exp>/summary.csv` and `by_hops.csv` (`--results-dir` to point elsewhere).
Error bars are mean ± 95% CI across seeds using Student's t (n = 5 is too small for 1.96).
SPR is always blue with circles and KAR orange with squares. In E2 the intervals cover
traffic randomness on a fixed topology only (see Main results).

| Figure | Content |
|---|---|
| `e3_gain_vs_bridge_share` | **headline:** KAR's relative outage reduction vs max bridge share, one dot per topology, colored by degree; E3 (left) and E3b (right) |
| `e1_outage_vs_hops`, `e1_latency_vs_hops` | chain: outage and mean latency (delivered requests) vs hop count in ~6 bins (1–5, …, 26–31), one line per ρ |
| `e2_outage_vs_rho`, `e2_throughput_vs_rho`, `e2_utilization_vs_rho` | both meshes side by side, `rho_sat` marked; throughput with the offered-load line |
| `e2_latency_vs_rho` | p95 latency above outage (latency alone hides survivorship bias) |
| `e2_paired_gain` | KAR − SPR outage, paired by seed, with 95% CI |
| `e3_outage_vs_degree`, `e3_throughput_vs_degree`, `e3_kar_gain_vs_degree` | density sweep vs achieved mean degree (fixed ρ) |
| `e3b_outage_vs_degree`, `e3b_throughput_vs_degree`, `e3b_kar_gain_vs_degree` | the same at a fixed arrival rate (E3b) |
| `e4_secoqc_bars` | SECOQC outage and throughput by ρ |
| `topo_mesh0_vs_diverse` | both E2 meshes; edge width ∝ SPR pair share, bridges highlighted |

## Mesh selection (E2)

`python scripts/select_mesh.py` scans topology seeds 0–49 of the default mesh and selects
the **lowest seed whose largest bridge carries at most 15% of node pairs**
(`max_bridge_share <= 0.15`). The rule uses topology only (bridges and SPR pair shares),
never routing results, and was applied before any E2 results were examined. It selects
seed 8.

**Caveat: the diverse mesh is atypical.** Only 2 of 50 seeds (8 and 12) meet the rule;
degree-4 random geometric meshes usually contain bridges that carry a large share of
traffic (in 33 of 50 seeds SPR's busiest link is itself a bridge). E2's `diverse` results
show what KAR can do when path diversity exists, not what it does on a typical degree-4
mesh. E3 and E3b sample topologies at random and are the better guide to typical behaviour.

## Known limitations

- p95 latency of delivered requests jumps to ≈ the 5 s timeout as soon as any queueing
  occurs, so it separates "no waiting" from "waiting" but little else; latency is always
  shown next to outage because delivered-request latency suffers survivorship bias.
- KAR sees exact, instantaneous key levels (no signalling delay or cost); SECOQC's link-state
  updates themselves consumed key (Peev Sec. 4.2).
- Waiting requests are not served FIFO (A5); the small pool capacity is a deliberate choice.
- Python 3.11 compatibility is checked by grammar only; tests run on 3.15.

## Repository layout

```
configs/default.yaml           default parameters
data/secoqc.yaml               SECOQC topology and per-link rates (Peev 2009), read-only
src/qkdsim/                    simulator package
  config.py links.py topologies.py routing.py traffic.py simulator.py metrics.py run.py
  experiment.py                sweep runner (used by experiments/run_experiment.py)
  analysis.py                  CI / aggregation helpers for plots and tables
experiments/configs/           E1-E4 experiment configs
experiments/run_experiment.py  sweep CLI
scripts/                       select_mesh.py, make_plots.py, make_tables.py
results/                       run outputs (git-ignored)
figures/                       figures and tables.md
tests/                         pytest suite
PLAN.md                        design plan and addenda A-E
```
