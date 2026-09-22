#!/usr/bin/env python
"""Persistent SSH daemon for the MetaX C500 dev box.

Holds ONE long-lived paramiko transport; each command runs as an exec channel
on it (no per-command handshake, no PTY/marker hacks). Clients talk over
127.0.0.1:17653 with length-prefixed JSON:
    request:  {"cmd": "...", "timeout": 300}
    response: {"rc": int, "out": "...", "err": "..."}
Auto-reconnects when the gateway drops the transport.
"""
import json
import os
import socketserver
import struct
import threading
import time

import paramiko

HOST = os.environ.get("RSH_HOST", "140.207.205.81")
PORT = int(os.environ.get("RSH_PORT", "32222"))
USER = os.environ.get("RSH_USER", "root+vm-vFIF4p6ezAhVkQWb")
PW = os.environ.get("RSH_PASS", "GPU123456789")
LISTEN = ("127.0.0.1", 17653)

ENV_PREFIX = (
    "export PATH=/opt/conda/bin:$PATH MACA_PATH=/opt/maca MACA_HOME=/opt/maca "
    "LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/lib64 VLLM_PLUGINS=fl; "
)


class SSHPool:
    def __init__(self):
        self.lock = threading.Lock()
        self.client = None
        self.connect()

    def connect(self):
        for attempt in range(10):
            try:
                if self.client:
                    try:
                        self.client.close()
                    except Exception:
                        pass
                c = paramiko.SSHClient()
                c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                c.connect(HOST, port=PORT, username=USER, password=PW,
                          timeout=30, allow_agent=False, look_for_keys=False)
                c.get_transport().set_keepalive(30)
                self.client = c
                print("[rserver] connected", flush=True)
                return
            except Exception as e:
                print(f"[rserver] connect attempt {attempt} failed: {e}", flush=True)
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("cannot connect")

    def run(self, cmd, timeout):
        with self.lock:
            for attempt in range(3):
                try:
                    _, out, err = self.client.exec_command(
                        ENV_PREFIX + cmd, timeout=timeout + 30)
                    rc = out.channel.recv_exit_status()
                    return rc, out.read().decode(errors="replace"), \
                        err.read().decode(errors="replace")
                except (OSError, EOFError, paramiko.SSHException) as e:
                    print(f"[rserver] exec failed ({type(e).__name__}: {e!r}), reconnecting", flush=True)
                    try:
                        self.connect()
                    except Exception as e2:
                        print(f"[rserver] reconnect failed: {type(e2).__name__} {e2!r}", flush=True)
                    time.sleep(5)
            return -1, "", "ERROR: command failed after reconnects"


pool = SSHPool()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            hdr = b""
            while len(hdr) < 8:
                chunk = self.request.recv(8 - len(hdr))
                if not chunk:
                    return
                hdr += chunk
            (n,) = struct.unpack(">Q", hdr)
            data = b""
            while len(data) < n:
                chunk = self.request.recv(min(65536, n - len(data)))
                if not chunk:
                    return
                data += chunk
            job = json.loads(data.decode())
            rc, out, err = pool.run(job["cmd"], int(job.get("timeout", 300)))
            resp = json.dumps({"rc": rc, "out": out, "err": err}).encode()
            self.request.sendall(struct.pack(">Q", len(resp)) + resp)
        except (ConnectionError, OSError):
            pass


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    srv = Server(LISTEN, Handler)
    print(f"[rserver] listening on {LISTEN}", flush=True)
    srv.serve_forever()
