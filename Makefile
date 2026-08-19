.PHONY: help run start stop restart status logs test

# Bare `make` shows the target list rather than silently running `run` (the
# first target otherwise wins by default), which is what someone typing
# `make` with no arguments actually wants to see first.
.DEFAULT_GOAL := help

VENV := .venv
UVICORN := $(VENV)/bin/uvicorn
PORT := 8801
LOG_DIR := logs
PID_FILE := $(LOG_DIR)/webapp.pid
LOG_FILE := $(LOG_DIR)/webapp.log
# Seconds to wait for uvicorn to bind the port, and for it to exit on stop.
# Importing the app costs well over ten seconds on a cold start, so the start
# budget is generous; a genuinely broken start still fails in about a second,
# because the wait loop gives up as soon as the process itself dies.
START_TIMEOUT := 60
STOP_TIMEOUT := 10

VENV_HINT := "No $(VENV) - run: python3 -m venv $(VENV) && $(VENV)/bin/pip install -r requirements.txt"

# Sets $pid and $running for use by start/stop/status. A PID file alone is not
# proof: the file may be stale and the OS may have recycled that PID for an
# unrelated process, so the command line is checked too - without this, `stop`
# would happily kill whatever now owns the number.
CHECK_RUNNING = pid=$$(cat $(PID_FILE) 2>/dev/null); \
	if [ -n "$$pid" ] && kill -0 $$pid 2>/dev/null && \
	   ps -p $$pid -o command= 2>/dev/null | grep -q 'uvicorn webapp.server:app'; then \
		running=1; \
	else \
		running=0; \
	fi

# PID of whatever is listening on $(PORT), regardless of whether `make start`
# put it there. Without this, `status`/`stop` only ever look at $(PID_FILE)
# and will report "Not running" while a server started some other way (a
# manual uvicorn invocation, an IDE launch config) is answering right there on
# the port - which is confusing and, for `stop`, would leave someone thinking
# they freed the port when they didn't.
PORT_HOLDER = if command -v lsof >/dev/null 2>&1; then \
		lsof -tiTCP:$(PORT) -sTCP:LISTEN 2>/dev/null | head -n1; \
	fi

# Runs in the foreground; Ctrl-C stops it. This is what README.md already
# documents, kept here so `make run` and `make start` share one venv check
# instead of drifting apart.
run: ## Run the server in the foreground (Ctrl-C to stop)
	@test -x $(UVICORN) || { echo $(VENV_HINT); exit 1; }
	$(UVICORN) webapp.server:app --port $(PORT)

# Runs in the background so `make start` returns immediately; `make stop` reads
# the PID back out of $(PID_FILE) to know what to kill. Returns only once the
# port is actually accepting connections, so `make start && curl ...` works.
# Readiness is probed over HTTP rather than by grepping $(LOG_FILE): uvicorn
# block-buffers its output when stdout is a file, so the startup banner can
# lag the open port by many seconds. A port already served by someone else is
# refused up front - otherwise that same probe would answer for the foreign
# server and report a success our own (bind-failed) process never achieved.
start: ## Start the server in the background
	@test -x $(UVICORN) || { echo $(VENV_HINT); exit 1; }
	@$(CHECK_RUNNING); \
	if [ $$running -eq 1 ]; then \
		echo "Already running on http://127.0.0.1:$(PORT) (PID $$pid)"; \
		exit 0; \
	fi; \
	ready() { \
		if command -v curl >/dev/null 2>&1; then \
			curl -s -o /dev/null "http://127.0.0.1:$(PORT)/"; \
		else \
			grep -q "Uvicorn running on" $(LOG_FILE) 2>/dev/null; \
		fi; \
	}; \
	if ready; then \
		echo "Port $(PORT) is already served by another process - refusing to start."; \
		echo "Stop that server first, or start this one elsewhere: make start PORT=<other>"; \
		exit 1; \
	fi; \
	mkdir -p $(LOG_DIR); \
	nohup $(UVICORN) webapp.server:app --port $(PORT) > $(LOG_FILE) 2>&1 & \
	pid=$$!; \
	echo $$pid > $(PID_FILE); \
	i=0; \
	while [ $$i -lt $(START_TIMEOUT) ]; do \
		if ! kill -0 $$pid 2>/dev/null; then \
			echo "Failed to start - last lines of $(LOG_FILE):"; \
			tail -n 15 $(LOG_FILE); \
			rm -f $(PID_FILE); \
			exit 1; \
		fi; \
		if ready; then \
			echo "Started on http://127.0.0.1:$(PORT) (PID $$pid); logs at $(LOG_FILE)"; \
			exit 0; \
		fi; \
		sleep 1; \
		i=$$((i + 1)); \
	done; \
	echo "Still not listening after $(START_TIMEOUT)s (PID $$pid) - check $(LOG_FILE), then 'make stop'"; \
	exit 1

# Waits for the process to actually exit rather than just signalling it, so a
# following `make start` cannot race the old server for the port or the log.
# Only ever kills the PID this Makefile itself launched (see CHECK_RUNNING) -
# a process holding $(PORT) that we didn't start is reported, not touched,
# since we have no way to know it's safe to kill.
stop: ## Stop the server started by 'make start'
	@$(CHECK_RUNNING); \
	if [ $$running -eq 0 ]; then \
		holder=$$($(PORT_HOLDER)); \
		rm -f $(PID_FILE); \
		if [ -n "$$holder" ]; then \
			echo "Not running via 'make start', but port $(PORT) is held by PID $$holder (started some other way) - leaving it alone."; \
			exit 1; \
		fi; \
		echo "Not running"; \
		exit 0; \
	fi; \
	kill $$pid; \
	i=0; \
	while [ $$i -lt $(STOP_TIMEOUT) ] && kill -0 $$pid 2>/dev/null; do \
		sleep 1; \
		i=$$((i + 1)); \
	done; \
	if kill -0 $$pid 2>/dev/null; then \
		echo "Did not exit after $(STOP_TIMEOUT)s - sending SIGKILL"; \
		kill -9 $$pid 2>/dev/null || true; \
		sleep 1; \
	fi; \
	rm -f $(PID_FILE); \
	echo "Stopped"

# Sub-makes rather than prerequisites, so stop always completes before start
# even under `make -j`.
restart: ## Stop then start the server
	@$(MAKE) stop
	@$(MAKE) start

status: ## Show whether the server is running
	@$(CHECK_RUNNING); \
	if [ $$running -eq 1 ]; then \
		echo "Running on http://127.0.0.1:$(PORT) (PID $$pid)"; \
		exit 0; \
	fi; \
	holder=$$($(PORT_HOLDER)); \
	if [ -n "$$holder" ]; then \
		echo "Not running via 'make start', but port $(PORT) is held by PID $$holder (started some other way)."; \
	else \
		echo "Not running"; \
	fi

logs: ## Follow the server's log output
	@if [ -f $(LOG_FILE) ]; then \
		tail -f $(LOG_FILE); \
	else \
		echo "No log file yet - start the server with 'make start' first"; \
	fi

test: ## Run the test suite
	@test -x $(VENV)/bin/pytest || { echo $(VENV_HINT); exit 1; }
	$(VENV)/bin/pytest -q

help: ## Show this help
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'
	@echo ""
	@echo "PORT defaults to $(PORT); override with e.g. 'make start PORT=8811'."
