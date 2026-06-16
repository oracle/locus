.PHONY: install research finance demo mesh test clean

PYTHON ?= python
PORT_RESEARCH ?= 8001
PORT_FINANCE  ?= 8002

install:
	$(PYTHON) -m pip install -r requirements.txt

research:
	PYTHONPATH=src $(PYTHON) -m a2a_mesh.research --port $(PORT_RESEARCH)

finance:
	PYTHONPATH=src $(PYTHON) -m a2a_mesh.finance --port $(PORT_FINANCE)

demo:
	PYTHONPATH=src $(PYTHON) -m a2a_mesh.orchestrator "Should I buy TSLA?"

mesh:
	@command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
	tmux new-session -d -s a2a-mesh -n research "make research"
	tmux split-window -t a2a-mesh -h "make finance"
	tmux split-window -t a2a-mesh -v "sleep 2 && make demo; read"
	tmux attach-session -t a2a-mesh

test:
	PYTHONPATH=src $(PYTHON) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
