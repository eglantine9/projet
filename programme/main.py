import sys
import os

# Ensure the current directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.ui.app import Application
except ImportError as e:
    print(f"Import Error: {e}")
    print("Ensure you are running this from the 'Assemblage' directory.")
    sys.exit(1)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
