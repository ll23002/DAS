#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
    echo "Python 3 no está instalado. Ejecute: sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Preparando el entorno por primera vez (puede tardar unos minutos)..."
    python3 -m venv .venv || { echo "Ejecute: sudo apt install python3-venv"; exit 1; }
fi

source .venv/bin/activate

if [ ! -f ".venv/.instalado" ] || [ requirements.txt -nt ".venv/.instalado" ]; then
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    touch .venv/.instalado
fi

LIB="tsw_rtm/c_engine/libwave2d.so"
if [ ! -f "$LIB" ]; then
    if ! command -v gcc >/dev/null 2>&1; then
        echo "Falta el compilador. Ejecute: sudo apt install build-essential"
        exit 1
    fi
    echo "Compilando el motor de simulación..."
    gcc -O3 -fopenmp -shared -fPIC -o "$LIB" tsw_rtm/c_engine/wave2d.c -lm
fi

python ejecutar.py
