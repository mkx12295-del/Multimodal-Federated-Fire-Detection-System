"""Copy the node to a Raspberry Pi over SSH and install it as a systemd service.

Credentials are never stored in the repository. Authentication uses your SSH
key (recommended: `ssh-copy-id pi@<host>`), or a password read from the
FIRENODE_SSH_PASSWORD environment variable.

    python deploy.py --host 192.168.1.50 --user pi
"""
import argparse
import os
import posixpath
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent
UPLOAD = ["firenode", "config", "models", "inference.py", "requirements-pi.txt",
          "deploy/firenode.service"]


def upload(sftp, local, remote):
    if local.is_dir():
        try:
            sftp.mkdir(remote)
        except OSError:
            pass  # already exists
        for child in local.iterdir():
            if child.name != "__pycache__":
                upload(sftp, child, posixpath.join(remote, child.name))
    else:
        sftp.put(str(local), remote)


def run(client, cmd):
    _, stdout, stderr = client.exec_command(cmd)
    for line in iter(stdout.readline, ""):
        print(line, end="")
    status = stdout.channel.recv_exit_status()
    if status != 0:
        raise RuntimeError(f"`{cmd}` failed ({status}): {stderr.read().decode().strip()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--user", default="pi")
    ap.add_argument("--dir", default="fire_project", help="target dir in the user's home")
    ap.add_argument("--no-install", action="store_true", help="copy files only")
    args = ap.parse_args()

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    client.connect(args.host, username=args.user,
                   password=os.environ.get("FIRENODE_SSH_PASSWORD"))
    try:
        target = f"/home/{args.user}/{args.dir}"
        run(client, f"mkdir -p {target}/deploy")
        sftp = client.open_sftp()
        for item in UPLOAD:
            local = ROOT / item
            if not local.exists():
                print(f"skip {item} (not found)")
                continue
            print(f"upload {item}")
            upload(sftp, local, posixpath.join(target, item))
        sftp.close()

        if not args.no_install:
            run(client, f"cd {target} && python3 -m venv --system-site-packages venv && "
                        f"venv/bin/pip install -r requirements-pi.txt")
            # The unit file uses the default paths; adjust it for other users.
            run(client, f"sed -e 's#/home/pi/fire_project#{target}#g' -e 's#User=pi#User={args.user}#' "
                        f"{target}/deploy/firenode.service | sudo tee /etc/systemd/system/firenode.service >/dev/null "
                        f"&& sudo systemctl daemon-reload && sudo systemctl enable --now firenode")
            print("Service installed: `sudo journalctl -u firenode -f` to watch it.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
