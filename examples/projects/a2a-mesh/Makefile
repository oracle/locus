.PHONY: install research finance demo mesh test clean

PORT_RESEARCH ?= 8001
PORT_FINANCE  ?= 8002

# The env is hatch-managed (see pyproject.toml). `sdk` installs the
# published SDK; use `hatch run sdk-local` instead to run against the
# repo this example ships in.
install:
	hatch env create && hatch run sdk

research:
	PORT_RESEARCH=$(PORT_RESEARCH) hatch run research

finance:
	PORT_FINANCE=$(PORT_FINANCE) hatch run finance

demo:
	hatch run demo

mesh:
	@command -v tmux >/dev/null || { echo "tmux not installed"; exit 1; }
	tmux new-session -d -s a2a-mesh -n research "make research"
	tmux split-window -t a2a-mesh -h "make finance"
	tmux split-window -t a2a-mesh -v "sleep 2 && make demo; read"
	tmux attach-session -t a2a-mesh

test:
	hatch run test

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
