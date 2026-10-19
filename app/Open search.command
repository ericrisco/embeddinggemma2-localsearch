#!/bin/zsh
# Double-click: starts the local search app and opens its window.
cd "$(dirname "$0")"
exec .venv/bin/python search.py
