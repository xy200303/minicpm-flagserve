#!/usr/bin/env python
"""Run a command on the MetaX C500 dev box over SSH (password auth).

Usage:
    python rsh.py "<command>" [timeout_seconds]
    python rsh.py --put <local> <remote>
    python rsh.py --get <remote> <local>
"""
import os
import sys
import paramiko

HOST = os.environ.get("RSH_HOST", "140.207.205.81")
PORT = int(os.environ.get("RSH_PORT", "32222"))
USER = os.environ.get("RSH_USER", "root+vm-vFIF4p6ezAhVkQWb")
PW = os.environ.get("RSH_PASS", "GPU123456789")


def client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              allow_agent=False, look_for_keys=False)
    return c


def main():
    args = sys.argv[1:]
    if args[0] == "--put":
        c = client()
        sftp = c.open_sftp()
        sftp.put(args[1], args[2])
        sftp.close(); c.close()
        print(f"uploaded {args[1]} -> {args[2]}")
        return
    if args[0] == "--get":
        c = client()
        _, out, _ = c.exec_command(f"base64 -w0 {args[1]}", timeout=120)
        rc = out.channel.recv_exit_status()
        if rc != 0:
            c.close()
            raise SystemExit(f"remote cat failed rc={rc}")
        import base64
        data = base64.b64decode(out.read().decode())
        c.close()
        with open(args[2], "wb") as f:
            f.write(data)
        print(f"downloaded {args[1]} -> {args[2]} ({len(data)} bytes)")
        return
    cmd = args[0]
    timeout = int(args[1]) if len(args) > 1 else 120
    c = client()
    _, out, err = c.exec_command(cmd, timeout=timeout)
    rc = out.channel.recv_exit_status()
    out.channel.settimeout(5)
    err.channel.settimeout(5)
    import socket as _socket
    try:
        sys.stdout.write(out.read().decode(errors="replace"))
    except (_socket.timeout, OSError):
        pass
    try:
        sys.stderr.write(err.read().decode(errors="replace"))
    except (_socket.timeout, OSError):
        pass
    c.close()
    sys.exit(rc)


if __name__ == "__main__":
    main()
