#!/usr/bin/env python3
"""
Launcher script for Streamlit Invoice Extraction UI
"""

import os
import sys
import subprocess
import click
import dotenv

dotenv.load_dotenv()

@click.group()
def cli():
    """Invoice Extraction Streamlit App Launcher"""
    pass

@cli.command()
@click.option('--port', default=8501, help='Port to run Streamlit on')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
def run(port, host):
    """Launch the Streamlit UI"""
    
    # Check if API key is set
    if not os.getenv("GEMINI_API_KEY"):
        click.echo("   Warning: GEMINI_API_KEY environment variable not set")
        click.echo("Please set it before running the app:")
        click.echo("export GEMINI_API_KEY='your-api-key'")
        if not click.confirm("Continue anyway?"):
            return
    
    # Launch Streamlit
    cmd = [
        sys.executable, 
        "-m", "streamlit", "run", 
        "streamlit_app.py",
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false"
    ]
    
    click.echo(f"🚀 Starting Streamlit on http://{host}:{port}")
    subprocess.run(cmd)

@cli.command()
def install():
    """Install required dependencies"""
    click.echo("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    click.echo("✅ Dependencies installed!")

if __name__ == "__main__":
    cli()