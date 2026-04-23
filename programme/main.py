import sys
import os
import subprocess

# Automatically restart using the local virtual environment if needed
_here = os.path.dirname(os.path.abspath(__file__))
venv_python = os.path.join(_here, "venv", "bin", "python3")

# Check if we are already running inside the project's venv
_in_venv = (
    hasattr(sys, "prefix")
    and os.path.realpath(sys.prefix) == os.path.realpath(os.path.join(_here, "venv"))
)

if not _in_venv and os.path.exists(venv_python):
    print("Switching to local virtual environment...")
    sys.exit(subprocess.call([venv_python, os.path.abspath(__file__)] + sys.argv[1:]))

# Ensure the current directory is in the path
sys.path.insert(0, _here)

try:
    from src.ui.app import Application
except ImportError as e:
    print(f"Import Error: {e}")
    print("Could not load application modules. Make sure dependencies are installed in programme/venv.")
    sys.exit(1)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
