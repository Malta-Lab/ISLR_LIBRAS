import subprocess

# Define the base folder and the remote server details
base_folder = '/B/zamin/MINDS/Sinalizador{:02d}/Canon/*.mp4'
remote_user = 'mussi'
remote_host = 'malta.lad.pucrs.br'
remote_path = '/mnt/F-SSD/MINDS'
password = '19920117'  # Replace with the actual password

# local_path = base_folder#.format(i)
# scp_command = f'sshpass -p {password} scp {local_path} {remote_user}@{remote_host}:{remote_path}'

# try:
#     print(f"Executing: {scp_command}")
#     result = subprocess.run(scp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     print(f"Output: {result.stdout.decode('utf-8')}")
#     print(f"Errors: {result.stderr.decode('utf-8')}")
# except subprocess.CalledProcessError as e:
#     print(f"Failed to transfer files from {local_path}")
#     print(f"Error: {e.stderr.decode('utf-8')}")

# Iterate over the folder numbers from 1 to 12
for i in range(1, 13):
    local_path = base_folder.format(i)
    scp_command = f'sshpass -p {password} scp {local_path} {remote_user}@{remote_host}:{remote_path}'
    
    try:
        print(f"Executing: {scp_command}")
        result = subprocess.run(scp_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Output: {result.stdout.decode('utf-8')}")
        print(f"Errors: {result.stderr.decode('utf-8')}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to transfer files from {local_path}")
        print(f"Error: {e.stderr.decode('utf-8')}")

print("File transfer completed.")