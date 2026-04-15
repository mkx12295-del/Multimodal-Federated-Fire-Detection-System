import paramiko
import os

hostname = '192.168.1.222'
username = 'spidey'
password = '4356'
target_dir = '/home/pi/fire_project'

print("Connecting to Raspberry Pi...")
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
try:
    client.connect(hostname, username=username, password=password)
    print("Connected.")

    print(f"Preparing directory: {target_dir}")
    stdin, stdout, stderr = client.exec_command(f'echo "{password}" | sudo -S mkdir -p {target_dir} && echo "{password}" | sudo -S chown spidey:spidey {target_dir}')
    status = stdout.channel.recv_exit_status()
    err = stderr.read().decode()
    if err and "password" not in err.lower():
        print(f"mkdir Error: {err}")

    print("\nSkipping SFTP upload (already complete). Retrying dependencies on Raspberry Pi...")
    cmd = 'pip install ultralytics xgboost bme680 opencv-python-headless --break-system-packages --retries 15 --timeout 120'
    # Let's execute pip install
    stdin, stdout, stderr = client.exec_command(cmd)
    
    # Print the output in real time
    for line in iter(stdout.readline, ""):
        try:
            print(line, end="")
        except UnicodeEncodeError:
            pass
    for line in iter(stderr.readline, ""):
        try:
            print(line, end="")
        except UnicodeEncodeError:
            pass
    
    print("\nVerifying setup...")
    stdin, stdout, stderr = client.exec_command(f'ls -l {target_dir} && python --version')
    print("Verification Output:")
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print("Verification Errors:", err)

    print("\nMigration Mission Accomplished!")
except Exception as e:
    print(f"Error during migration: {e}")
finally:
    client.close()
