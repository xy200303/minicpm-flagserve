#!/usr/bin/env python
"""Client for rserver.py — run a command on the dev box via the persistent shell.

Usage:
    python rcli.py "<command>" [timeout_seconds]
    python rcli.py --get <remote> <local>
    python rcli.py --put <local> <remote>
Exit code = remote command exit code.
"""
import base64
import json
import re
import socket
import struct
import sys

ADDR = ("127.0.0.1", 17653)
_MSYS = re.compile(r"^[A-Za-z]:/.*?/Git(/.*)$")


def unmangle(a):
    m = _MSYS.match(a)
    return m.group(1) if m else a


def call(job):
    s = socket.create_connection(ADDR, timeout=30)
    s.settimeout(int(job.get("timeout", 300)) + 90)
    data = json.dumps(job).encode()
    s.sendall(struct.pack(">Q", len(data)) + data)
    hdr = b""
    while len(hdr) < 8:
        hdr += s.recv(8 - len(hdr))
    (n,) = struct.unpack(">Q", hdr)
    resp = b""
    while len(resp) < n:
        chunk = s.recv(min(65536, n - len(resp)))
        if not chunk:
            break
        resp += chunk
    s.close()
    return json.loads(resp.decode())


def main():
    args = [unmangle(a) for a in sys.argv[1:]]
    if args and args[0] == "--get":
        r = call({"cmd": f"base64 -w0 '{args[1]}'", "timeout": 300})
        if r["rc"] != 0:
            sys.stderr.write(r["out"])
            sys.exit(r["rc"])
        with open(args[2], "wb") as f:
            f.write(base64.b64decode(r["out"]))
        print(f"downloaded {args[1]} -> {args[2]}")
        return
    if args and args[0] == "--put":
        with open(args[1], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        r = call({"cmd": f"echo {b64} | base64 -d > '{args[2]}'", "timeout": 300})
        if r["rc"] != 0:
            sys.stderr.write(r["out"])
            sys.exit(r["rc"])
        print(f"uploaded {args[1]} -> {args[2]}")
        return
    timeout = int(args[1]) if len(args) > 1 else 300
    r = call({"cmd": args[0], "timeout": timeout})
    sys.stdout.write(r["out"])
    sys.exit(r["rc"])


if __name__ == "__main__":
    main()
