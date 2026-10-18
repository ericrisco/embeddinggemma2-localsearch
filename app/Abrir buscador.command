#!/bin/zsh
# Doble clic: arranca el buscador local y abre su ventana.
cd "$(dirname "$0")"
exec .venv/bin/python buscador.py
