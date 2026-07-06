import os
import sys

# Make sure `import app...` resolves regardless of where pytest is invoked
# from -- this file's own directory is the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
