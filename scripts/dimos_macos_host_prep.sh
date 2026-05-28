#!/usr/bin/env bash
set -euo pipefail

# Prepare macOS networking limits for local DimOS LCM replay/simulation.
# This requires an interactive sudo session. It does not start DimOS, MuJoCo,
# replay, MCP, or any robot process.

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This host-prep script is macOS-only." >&2
  exit 2
fi

echo "Preparing macOS host settings for DimOS LCM replay/simulation."
echo "This will ask for sudo and run route/sysctl commands only."

sudo -v

sudo route delete -net 224.0.0.0/4 >/dev/null 2>&1 || true
sudo route add -net 224.0.0.0/4 -interface lo0

sudo sysctl -w kern.ipc.maxsockbuf=67108864
sudo sysctl -w net.inet.udp.recvspace=67108864
sudo sysctl -w net.inet.udp.maxdgram=67108864

echo
echo "Resulting multicast route:"
netstat -nr | awk '/224\\.0\\.0(\\.0)?\\/4/ {print}'

echo
echo "LCM socket settings:"
sysctl kern.ipc.maxsockbuf net.inet.udp.recvspace net.inet.udp.maxdgram

echo
echo "Host prep complete. Next:"
echo "  make dimos-replay-smoke"
echo "  make dimos-sim-smoke"
