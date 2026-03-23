from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import uvicorn


def main() -> None:
    uvicorn.run("geochemad.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
