#!/bin/sh
set -eu

# Patchright remains headed. Xvfb supplies the Linux display and noVNC exposes
# it at localhost:6080 so selector capture and login recording stay usable.
Xvfb "${DISPLAY:-:99}" -screen 0 1440x900x24 -nolisten tcp &
sleep 1
x11vnc -display "${DISPLAY:-:99}" -forever -shared -nopw -rfbport 5900 -localhost &
websockify --web=/usr/share/novnc 6080 localhost:5900 &

exec python -m app.run
