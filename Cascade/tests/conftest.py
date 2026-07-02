import sys
from pathlib import Path

# Make `import cascade` work regardless of the caller's cwd, without requiring
# an installed package / pyproject.toml for this pass.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
