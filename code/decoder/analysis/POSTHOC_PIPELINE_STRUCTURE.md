# Posthoc Analysis Pipeline Structure (Proposed)

This proposal is focused only on the group-level posthoc pipeline around `group_analysis.m` and its helper functions.

## Goals

1. Make it easy to add a **new analysis** without editing many files.
2. Make it easy to add a **new plot** without coupling plotting to computation.
3. Keep each step deterministic, cacheable, and testable in isolation.

---

## Current pain points in `group_analysis.m`

- One script mixes configuration, compute, stats, plotting, and figure export.
- Subject lists and paths are hard-coded in multiple blocks.
- Figures are exported from "all open figures", which is fragile.
- Some stats are manually typed in the script instead of derived from data.

---

## Recommended architecture

Use a small, explicit orchestration flow:

1. **Config** (one place for subjects, roots, flags)
2. **Load/Assemble** (turn files into standardized in-memory structs/tables)
3. **Compute metrics** (pure functions; no plotting)
4. **Run stats** (pure functions returning result structs/tables)
5. **Plot** (functions that accept computed outputs and return figure handles)
6. **Export** (save only figures explicitly returned by plot functions)

A practical folder layout:

```text
code/decoder/analysis/
  group_analysis.m                % thin entry point (or call run_group_posthoc)
  run_group_posthoc.m             % orchestration only
  +cfg/
    make_posthoc_config.m         % subjects, sessions, paths, switches
  +io/
    load_group_cache.m
    load_subject_cache.m
    list_cache_files.m
  +metrics/
    compute_rt_effects.m
    compute_stroop_effects.m
    compute_runwise_metrics.m
    compute_auc_auprc_per_session.m
    computePdR2_pairDiffTopos.m
  +stats/
    run_rt_models.m
    run_stroop_models.m
    run_group_interactions.m
  +plots/
    plot_rt_stroop_prepost.m
    plot_pd_topos.m
    plot_pd_po78.m
  +pipeline/
    register_analyses.m           % list of analysis modules to run
    run_registered_analyses.m
  +util/
    save_figures.m
    ensure_dir.m
    stamp_now.m
```

---

## Data contract (important for extensibility)

Keep one canonical `GROUP`-level struct so new modules can plug in consistently.

Suggested top-level contract:

- `GROUP.meta`: subjects, groups, session map, timestamps.
- `GROUP.raw`: minimally processed loaded content from cache/files.
- `GROUP.metrics.<module_name>`: numeric outputs/tables from each compute module.
- `GROUP.stats.<module_name>`: model outputs, p-values, effect sizes.
- `GROUP.figures.<module_name>`: figure handles and labels.

If every module reads from this contract and writes to its own namespace, you can add modules without breaking others.

---

## Analysis registry pattern

Instead of hard-coding each block in one script, define a registry of modules.

Each module entry should include:

- `name`: unique key
- `enabled`: boolean
- `compute_fn`: `@(ctx) ...`
- `stats_fn`: optional
- `plot_fn`: optional
- `depends_on`: list of module names

Then `run_registered_analyses` executes modules in dependency order and stores outputs in `ctx`.

This gives you easy extension: add one module file + one registry entry.

---

## Recommended conventions

- **No plotting inside compute functions**.
- **No file I/O inside stats functions** except optional output save wrappers.
- Return `table` where possible for easier downstream stats.
- Use explicit inputs/outputs; avoid relying on workspace state.
- Keep subject/group definitions only in config.

---

## How to add a new analysis quickly

1. Create `+metrics/compute_<name>.m` returning a struct/table.
2. (Optional) Create `+stats/run_<name>_stats.m`.
3. (Optional) Create `+plots/plot_<name>.m` returning figure handle(s).
4. Add one entry in `+pipeline/register_analyses.m`.
5. Run `run_group_posthoc` with module enabled.

That should be the only change set in most cases.

---

## Migration plan from current code

1. Keep existing helpers (`compute_rt_effects`, `run_stroop_group_anovas`, etc.) as-is.
2. Introduce new orchestrator and registry; call existing helpers from module wrappers.
3. Move hard-coded subject lists/paths into config.
4. Replace "export all open figures" with explicit figure lists.
5. After parity, deprecate ad-hoc blocks in `group_analysis.m`.

---

## Minimal file responsibilities

- `group_analysis.m`: one-liner entry (`cfg = ...; out = run_group_posthoc(cfg);`).
- `run_group_posthoc.m`: controls phase order and logging only.
- module functions: do one thing, return one namespace output.
- `save_figures.m`: centralized PDF/FIG export behavior.

This keeps the pipeline maintainable as you add analyses/plots over time.
