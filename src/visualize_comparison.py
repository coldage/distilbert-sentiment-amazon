"""
Script to visualize model comparison results from JSON file.
Run this after training to generate graphs and reports.

Usage:
    python src/visualize_comparison.py
    python src/visualize_comparison.py --json output/model_comparison.json

TODO the pdf is a bit unnecessary and the html is just overkill lol
"""

import os
import rootutils
import argparse
from utils.visualize_results import visualize_results

# Setup root
root_path = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.chdir(root_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model comparison results")
    parser.add_argument("--json", type=str, default="output/model_comparison.json",
                       help="Path to JSON results file")
    parser.add_argument("--output", type=str, default="output",
                       help="Output directory for visualizations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json):
        print(f"Error: Results file not found: {args.json}")
        print("Please run training first to generate results.")
        exit(1)
    
    print(f"Loading results from {args.json}...")
    visualize_results(args.json, args.output)

