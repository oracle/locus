.PHONY: install intel triage demo mesh test clean

PORT_INTEL  ?= 8001
PORT_TRIAGE ?= 8002

# The env is hatch-managed (see pyproject.toml). `sdk` installs the
# published SDK; use `hatch run sdk-local` instead to run against the
# repo this example ships in.
install:
	hatch env create && hatch run sdk

intel:
	PORT_INTEL=$(PORT_INTEL) hatch run intel

triage:
	PORT_TRIAGE=$(PORT_TRIAGE) hatch run triage

demo:
	hatch run demo

mesh:
	@command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
	tmux new-session -d -s a2a-mesh -n intel "make intel"
	tmux split-window -t a2a-mesh -h "make triage"
	tmux split-window -t a2a-mesh -v "sleep 2 && make demo; read"
	tmux attach-session -t a2a-mesh

test:
	hatch run test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
