Git / GitHub root under ABR2synapse
===================================

Ignore rules: edit `ABR2synapse/.gitignore` (canonical), then run
  cp ABR2synapse/.gitignore .gitignore
so the monorepo root copy stays in sync for Git.


Goal: the GitHub repository root should match the *contents* of this folder
(one checkout shows `utils/`, `figures/`, notebooks, etc. at the top level,
not `ABR2synapse/...` nested inside a larger tree).

Current layout (transitional)
-----------------------------
You may have a *parent* folder (e.g. Practicum) that contains both:
  - project files at the parent root, and
  - this `ABR2synapse/` subtree (sometimes with its own `figures/cache/` exports).

Python cache resolution
-----------------------
`utils/benchmark_metrics.resolve_cache_file(...)` loads each cache file from
`<repo>/figures/cache` first, then `<repo>/ABR2synapse/figures/cache`, so split
exports still work until you finish the move.

Recommended git migration (outline only; review before running)
--------------------------------------------------------------
1. Commit or stash all work.

2. Move *authoritative* project files (notebooks, `utils/`, data dirs, etc.)
   into `ABR2synapse/` so nothing important remains only at the parent root.
   Resolve duplicate names (e.g. two `utils/` trees) by keeping one canonical
   copy and removing the other after diffing.

3. Promote this folder to the repository root on disk (pick one approach):

   A) `git filter-repo` (install `git-filter-repo` first), from the *current*
      repo root, to keep only the `ABR2synapse/` tree as the new root (rewrites
      history; learn the exact flags from `git filter-repo -h` and test on a
      clone).

   B) Create a *new* repository: `cd ABR2synapse && git init`, add remote, push
      (simplest; does not preserve old history unless you use grafts/subtree).

4. Point GitHub’s default branch at the new repository and archive or delete the
   old remote if obsolete.

5. Re-open the IDE / Jupyter with **workspace root = the new repo root** so
   `Path.cwd()` matches `utils.benchmark_metrics.repo_root()`.
