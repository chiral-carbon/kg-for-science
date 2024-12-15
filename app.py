import os
import sys

# Add the project root directory to Python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.run_db_interface import create_demo

demo = create_demo()

if __name__ == "__main__":
    demo.launch()