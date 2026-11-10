"""Tiny RPC between the "robot + camera driver" (Isaac Sim) and the scanning brain.

Only stdlib + JSON + raw bytes cross the wire, so the two sides can use different
Python / numpy versions.  The protocol deliberately mirrors what a real robot
stack exposes: joint encoders, joint commands, camera frames.  There is no call
that returns simulator ground truth.
"""
from __future__ import annotations

import json
from multiprocessing.connection import Client, Listener

AUTHKEY = b"arm-scan-trellis"
# Commands the brain may send.  Anything else is rejected by the server.
COMMANDS = {
    "hello",        # -> robot model name, dof names, joint limits, camera resolution
    "get_joints",   # -> encoder readings (7 arm joints)
    "move",         # joint-space waypoints, executed with a smooth profile
    "capture",      # -> wrist RGB frame (PNG bytes) + encoder readings at exposure
    "annotate",     # brain pushes a status line / overlay picture into the recording
    "idle",         # keep recording for N seconds without moving (e.g. while computing)
    "shutdown",
}


def send(conn, header: dict, blob: bytes | None = None) -> None:
    header = dict(header)
    header["has_blob"] = blob is not None
    conn.send_bytes(json.dumps(header).encode())
    if blob is not None:
        conn.send_bytes(blob)


def recv(conn) -> tuple[dict, bytes | None]:
    header = json.loads(conn.recv_bytes().decode())
    blob = conn.recv_bytes() if header.get("has_blob") else None
    return header, blob


def listen(port: int) -> Listener:
    return Listener(("127.0.0.1", port), authkey=AUTHKEY)


def connect(port: int):
    return Client(("127.0.0.1", port), authkey=AUTHKEY)
