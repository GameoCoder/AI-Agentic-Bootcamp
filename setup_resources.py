"""
setup_resources.py
──────────────────
Downloads the Indic NLP Library resources from GitHub.
Run this ONCE after `pip install -r requirements.txt`.

Usage:
    python setup_resources.py
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import urllib.request

RESOURCES_URL = (
    "https://github.com/anoopkunchukuttan/indic_nlp_resources/archive/refs/heads/master.zip"
)
RESOURCES_DIR = Path(__file__).parent / "indic_nlp_resources"
ZIP_PATH = Path(__file__).parent / "indic_nlp_resources.zip"


def download_resources():
    if RESOURCES_DIR.exists():
        print(f"✅ Indic NLP resources already present at: {RESOURCES_DIR}")
        return

    print(f"⬇  Downloading Indic NLP resources from GitHub…")
    print(f"   URL: {RESOURCES_URL}")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r   {pct}% ({downloaded // 1024} KB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(RESOURCES_URL, ZIP_PATH, reporthook=_progress)
        print()
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("   Manual install: git clone https://github.com/anoopkunchukuttan/indic_nlp_resources")
        sys.exit(1)

    print("📦 Extracting…")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(Path(__file__).parent)

    # Rename extracted folder
    extracted = Path(__file__).parent / "indic_nlp_resources-master"
    if extracted.exists():
        extracted.rename(RESOURCES_DIR)

    ZIP_PATH.unlink(missing_ok=True)
    print(f"✅ Resources extracted → {RESOURCES_DIR}")


def set_env_variable():
    """Tell indic-nlp-library where resources are."""
    env_path = RESOURCES_DIR / "indic_nlp_resources"
    if not env_path.exists():
        # Some versions store directly in the root
        env_path = RESOURCES_DIR

    os.environ["INDIC_RESOURCES_PATH"] = str(env_path)

    # Also write to .env so it persists
    dot_env = Path(__file__).parent / ".env"
    line = f'INDIC_RESOURCES_PATH="{env_path}"\n'
    if dot_env.exists():
        content = dot_env.read_text()
        if "INDIC_RESOURCES_PATH" not in content:
            with open(dot_env, "a") as f:
                f.write(line)
            print(f"✅ Added INDIC_RESOURCES_PATH to .env")
    else:
        dot_env.write_text(line)
        print(f"✅ Created .env with INDIC_RESOURCES_PATH")

    print(f"   INDIC_RESOURCES_PATH={env_path}")


def verify_install():
    """Quick sanity-check that the library can tokenise."""
    try:
        from indicnlp import common
        from indicnlp.tokenize import indic_tokenize
        common.set_resources_path(str(RESOURCES_DIR / "indic_nlp_resources"))
        tokens = indic_tokenize.trivial_tokenize("भारत एक महान देश है", lang="hi")
        print(f"✅ Tokenization test passed: {tokens}")
    except Exception as e:
        print(f"⚠  Tokenization test failed (whitespace fallback will be used): {e}")


if __name__ == "__main__":
    download_resources()
    set_env_variable()
    verify_install()
    print("\n✅ Setup complete! You can now run the project.")
