import os
import paramiko
from dotenv import load_dotenv

load_dotenv()

def ensure_remote_dir_exists(sftp, remote_dir):
    """
    Recursively create directories on the SFTP server if they don't exist.
    """
    dirs = remote_dir.strip("/").split("/")
    current_path = ""
    for directory in dirs:
        current_path += f"/{directory}"
        print(f"Checking if directory exists: {current_path}")
        try:
            sftp.stat(current_path)
        except FileNotFoundError:
            sftp.mkdir(current_path)

import socket # Import socket for specific error handling

def upload_to_sftp(file_content: bytes, filename: str, remote_dir: str) -> str:
    SFTP_HOST = os.getenv("SFTP_HOST")
    SFTP_PORT = os.getenv("SFTP_PORT")
    SFTP_USERNAME = os.getenv("SFTP_USERNAME")
    SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")

    if not all([SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD]):
        raise ValueError("SFTP environment variables (SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD) are not set.")

    remote_path = f"{remote_dir}/{filename}"

    try:
        transport = paramiko.Transport((SFTP_HOST, int(SFTP_PORT)))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            ensure_remote_dir_exists(sftp, remote_dir)
        except (paramiko.SSHException, socket.error) as e:
            raise RuntimeError(f"SFTP directory creation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during SFTP directory creation: {e}")

        try:
            with sftp.file(remote_path, 'wb') as remote_file:
                remote_file.write(file_content)
        except (paramiko.SSHException, socket.error) as e:
            raise RuntimeError(f"SFTP upload failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during SFTP upload: {e}")

        sftp.close()
        transport.close()

        return remote_path
    except (paramiko.SSHException, socket.error) as e:
        raise RuntimeError(f"SFTP connection failed: {e}. Please check SFTP server status, host, port, and network connectivity.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during SFTP operation: {e}")

def download_from_sftp(remote_filepath: str, local_filepath: str) -> None:
    SFTP_HOST = os.getenv("SFTP_HOST")
    SFTP_PORT = os.getenv("SFTP_PORT")
    SFTP_USERNAME = os.getenv("SFTP_USERNAME")
    SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")

    if not all([SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD]):
        raise ValueError("SFTP environment variables (SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD) are not set.")

    try:
        transport = paramiko.Transport((SFTP_HOST, int(SFTP_PORT)))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            sftp.get(remote_filepath, local_filepath)
        except (paramiko.SSHException, socket.error) as e:
            raise RuntimeError(f"SFTP download failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during SFTP download: {e}")

        sftp.close()
        transport.close()

    except (paramiko.SSHException, socket.error) as e:
        raise RuntimeError(f"SFTP connection failed: {e}. Please check SFTP server status, host, port, and network connectivity.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during SFTP operation: {e}")

def read_file_from_sftp(remote_filepath: str) -> bytes:
    SFTP_HOST = os.getenv("SFTP_HOST")
    SFTP_PORT = os.getenv("SFTP_PORT")
    SFTP_USERNAME = os.getenv("SFTP_USERNAME")
    SFTP_PASSWORD = os.getenv("SFTP_PASSWORD")

    if not all([SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD]):
        raise ValueError("SFTP environment variables (SFTP_HOST, SFTP_PORT, SFTP_USERNAME, SFTP_PASSWORD) are not set.")

    try:
        transport = paramiko.Transport((SFTP_HOST, int(SFTP_PORT)))
        transport.connect(username=SFTP_USERNAME, password=SFTP_PASSWORD)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            with sftp.file(remote_filepath, "rb") as remote_file:
                content = remote_file.read()
        except (paramiko.SSHException, socket.error) as e:
            raise RuntimeError(f"SFTP read failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during SFTP read: {e}")
        finally:
            sftp.close()
            transport.close()

        return content

    except (paramiko.SSHException, socket.error) as e:
        raise RuntimeError(f"SFTP connection failed: {e}. Please check SFTP server status, host, port, and network connectivity.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during SFTP operation: {e}")
