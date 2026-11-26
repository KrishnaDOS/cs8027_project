# Live demo: run the diverse-mesh config with SPR, then KAR, and print a side-by-side comparison.
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
# Uses the project's .venv if present, otherwise `python` on PATH.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }
$config = Join-Path $root "experiments\configs\demo_diverse.yaml"

# ASCII only below: PowerShell 5.1 pipes text to native programs as ASCII.
$code = @'
import sys, time, warnings
from qkdsim.config import load_config
from qkdsim.metrics import summarize
from qkdsim.simulator import run_simulation

warnings.simplefilter("ignore")
path = sys.argv[1]
results = {}
for router in ("spr", "kar"):
    cfg = load_config(path, overrides={"routing": {"strategy": router}})
    print(f"running {router.upper()} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    result = run_simulation(cfg, cfg.simulation.seeds[0])
    results[router] = summarize(result, cfg)
    print(f"{time.perf_counter() - t0:.1f} s")

s, k = results["spr"], results["kar"]
print()
print(f"diverse mesh (topology seed {s['topology_seed']}, {s['n_bridges']} bridges, "
      f"max bridge share {s['max_bridge_share']:.3f}), rho = {s['offered_load']:g}, "
      f"rho_sat = {s['rho_sat']:.3f}, {s['measured_duration_s']:g} s measured")
print(f"{'':14}{'SPR':>12}{'KAR':>12}")
print(f"{'outage':14}{s['outage_probability']:12.3f}{k['outage_probability']:12.3f}")
print(f"{'throughput':14}{s['delivered_throughput_bps']:12,.0f}{k['delivered_throughput_bps']:12,.0f}  bit/s")
print(f"{'mean hops':14}{s['mean_hops']:12.2f}{k['mean_hops']:12.2f}")
'@

$timer = [Diagnostics.Stopwatch]::StartNew()
$code | & $python - $config
if ($LASTEXITCODE -ne 0) { throw "demo failed (exit code $LASTEXITCODE)" }
"`ntotal wall-clock: {0:N1} s" -f $timer.Elapsed.TotalSeconds
