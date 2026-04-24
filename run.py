import subprocess
import sys

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("Training model...")
subprocess.run([sys.executable, "src/train.py"])

print("Starting n8n...")
subprocess.run(["npx", "n8n"])