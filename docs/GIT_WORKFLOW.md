# Team Git workflow

## Branches
| Branch | Owner | Folders the owner may change freely |
|---|---|---|
| `main` | team rule: changes go in through PRs only (GitHub can't enforce this on a free private repo) | – |
| `feature/data-pipeline` | Prithvish | `src/pathsnap/data/`, `candidates/`, `features/`, `results/eda/`, `results/candidates/` |
| `feature/ml-models` | Krishna | `src/pathsnap/models/`, `results/models/`, `results/optuna/`, `configs/models*.yaml` |
| `feature/hmm-decoder` | Pulkit | `src/pathsnap/labels/`, `src/pathsnap/hmm/`, `results/hmm/` |
| `feature/backend-frontend` | Priyanka | `backend/`, `frontend/`, `src/pathsnap/eval/`, `src/pathsnap/viz/folium_maps.py`, `docs/DEPLOY.md` |

Shared files (`README.md`, `requirements.txt`, `src/pathsnap/pipeline.py`, `config.py`) can be changed in any PR, but **say so in the PR description** and tag the affected owner.

## First-time setup (each member)

```bash
winget install Git.Git
```
```bash
git clone https://github.com/GitGud-User/pathsnap.git
```
```bash
cd pathsnap
```
```bash
git config user.name "Your Name"
```
```bash
git config user.email "your-github-email@example.com"
```
```bash
git checkout feature/<your-branch>
```

Then follow the README setup (venv, `pip install -r requirements.txt`, `pip install -e .`) and copy the dataset into `data/`.

## Daily loop
```bash
git pull origin main
```
```bash
git merge main
```
… work, run `pytest -q` …
```bash
git add <files>
```
```bash
git commit -m "data: add STRtree candidate generator"
```
```bash
git push origin feature/<your-branch>
```

Open a PR into `main` on GitHub. **At least one other member must review it.** Use the review pairs from the roadmap:
- Krishna reviews Prithvish
- Pulkit reviews Krishna
- Prithvish reviews Pulkit
- Pulkit reviews Priyanka

## Rules
1. Never commit anything under `data/`. `.gitignore` blocks it, so don't force-add with `git add -f`.
2. Never commit evaluator files, predictions for evaluator trips, or anything derived from evaluator routes.
3. Commit the **code that produced every result** together with the result file (CSV/PNG/JSON). The report must be reproducible from the repo.
4. Use small PRs with prefixed messages: `data:`, `models:`, `hmm:`, `api:`, `ui:`, `docs:`, `test:`.
5. `pytest -q` must pass before you open a PR.
6. Model files over ~50 MB: ask the team before committing them (GitHub's limit is 100 MB per file).
7. Repo is private during development and is **made public before submission** (Settings → Danger zone → Change visibility).
