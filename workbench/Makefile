# locus workbench — full stack: locus python A2A peers, Node BFF, vanilla TS web,
# Playwright e2e. Each `make` target runs in the foreground so you can chain
# them across panes / terminals.

.PHONY: install backend backend-research backend-finance bff web e2e clean

# === Backend (workbench FastAPI runner — the actual workbench backend) =======
# This is the runner the workbench UI talks to. Three tiers run side-by-side:
#   make backend   (uvicorn :8100)   | make bff   (Express :3101)
#   make web       (Vite :5173)
# The backend-research / backend-finance targets below are unrelated — they
# spin up the A2A mesh demo peers for notebook 34, not the workbench itself.
WORKBENCH_PORT ?= 8100

backend:
	cd backend && python -m uvicorn --app-dir . runner:app --port $(WORKBENCH_PORT)

# === A2A mesh demo peers (notebook 34 — separate from the workbench) =========
# Reuses the merged a2a-mesh project under examples/projects/a2a-mesh/src.
# Pass LOCUS_MODEL_PROVIDER=oci + the BOAT-OC1 env block from your shell to
# run against real OCI GenAI; defaults to MockModel.
BACKEND_PYTHONPATH = ../examples/projects/a2a-mesh/src:../examples:../src
RESEARCH_PORT ?= 8001
FINANCE_PORT  ?= 8012

backend-research:
	cd .. && PYTHONPATH=$(BACKEND_PYTHONPATH:..%=%) hatch -e default run python -m a2a_mesh.research --port $(RESEARCH_PORT)

backend-finance:
	cd .. && PYTHONPATH=$(BACKEND_PYTHONPATH:..%=%) hatch -e default run python -m a2a_mesh.finance --port $(FINANCE_PORT)

# === BFF (Node Express) =====================================================
bff:
	cd bff && npm run dev

# === Web (vanilla TypeScript via Vite) ======================================
web:
	cd web && npm run dev

# === E2E (Playwright) =======================================================
e2e:
	cd e2e && npm test

install:
	cd bff && npm install
	cd web && npm install
	cd e2e && npm install && npx playwright install chromium

clean:
	rm -rf bff/node_modules bff/dist web/node_modules web/dist e2e/node_modules e2e/test-results
