#!/usr/bin/env python3
"""
SFTP Client utility for uploading files and results to SFTP server
"""

import os
import json
import logging
import tempfile
from typing import Tuple

import paramiko
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)


class SFTPClient:
    def __init__(self):
        self.host = os.getenv("SFTP_HOST")
        self.port = int(os.getenv("SFTP_PORT", 22))
        self.username = os.getenv("SFTP_USERNAME")
        self.password = os.getenv("SFTP_PASSWORD")
        self.remote_dir = os.getenv("REMOTE_DIR", "/files/inHouseOCR")
        
        if not all([self.host, self.username, self.password]):
            raise ValueError("SFTP credentials not properly configured")
    
    def upload_file(self, file_content: bytes, filename: str, process_id: str) -> Tuple[str, str]:
        """Upload file to SFTP server with UUID folder structure"""
        try:
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to SFTP server
            ssh.connect(self.host, port=self.port, username=self.username, password=self.password)
            sftp = ssh.open_sftp()
            
            # Create remote directory path with UUID
            remote_folder = f"{self.remote_dir}/{process_id}"
            remote_file_path = f"{remote_folder}/{filename}"
            
            # Create directory if it doesn't exist
            try:
                sftp.stat(remote_folder)
            except FileNotFoundError:
                sftp.mkdir(remote_folder)
            
            # Upload file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                sftp.put(temp_file.name, remote_file_path)
                os.unlink(temp_file.name)
            
            sftp.close()
            ssh.close()
            
            logger.info(f"✅ File uploaded to SFTP: {remote_file_path}")
            return remote_file_path, remote_folder
            
        except Exception as e:
            logger.error(f"SFTP upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SFTP upload failed: {str(e)}"
            )
    
    def upload_json_result(self, json_data: dict, process_id: str, original_filename: str) -> str:
        """Upload JSON result to SFTP server"""
        try:
            # Create SSH client
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # Connect to SFTP server
            ssh.connect(self.host, port=self.port, username=self.username, password=self.password)
            sftp = ssh.open_sftp()
            
            # Create remote directory path with UUID
            remote_folder = f"{self.remote_dir}/{process_id}"
            result_filename = f"{original_filename}_results.json"
            remote_result_path = f"{remote_folder}/{result_filename}"
            
            # Ensure directory exists
            try:
                sftp.stat(remote_folder)
            except FileNotFoundError:
                sftp.mkdir(remote_folder)
            
            # Create temporary JSON file and upload
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                json.dump(json_data, temp_file, indent=2)
                temp_file.flush()
                sftp.put(temp_file.name, remote_result_path)
                os.unlink(temp_file.name)
            
            sftp.close()
            ssh.close()
            
            logger.info(f"✅ JSON result uploaded to SFTP: {remote_result_path}")
            return remote_result_path
            
        except Exception as e:
            logger.error(f"SFTP JSON upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"SFTP JSON upload failed: {str(e)}"
            )