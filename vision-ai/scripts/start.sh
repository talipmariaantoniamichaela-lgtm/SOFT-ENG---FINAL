#!/usr/bin/env bash
# Start the VISION AI API and Vite development server on macOS/Linux.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
backend_dir="${project_dir}/backend"
frontend_dir="${project_dir}/frontend"
python_bin="${PYTHON_BIN:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python 3 was not found. Set PYTHON_BIN to a Python 3 executable." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "npm was not found. Install Node.js 20 or later and try again." >&2
  exit 1
fi

cleanup() {
  trap - EXIT INT TERM
  kill "${backend_pid:-}" "${frontend_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

(
  cd "${backend_dir}"
  exec "${python_bin}" -m uvicorn app.main:app --host 127.0.0.1 --port 8080
) &
backend_pid=$!

(
  cd "${frontend_dir}"
  exec npm run dev -- --host 127.0.0.1 --port 5173
) &
frontend_pid=$!

echo "VISION AI is starting:"
echo "  Frontend: http://localhost:5173"
echo "  API docs: http://localhost:8080/api/docs"
echo "Press Ctrl-C to stop both servers."

wait "${backend_pid}" "${frontend_pid}"
