import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.run_db_interface import launch

if __name__ == "__main__":
    launch()
