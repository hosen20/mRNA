"""Small helpers so each notebook can run on its own in Google Colab.

The idea: you upload ONE notebook to Colab, and everything else (the code, the
saved results) lives in your Google Drive. Each notebook mounts Drive, finds
the project folder, and picks up where the last notebook left off.
"""

import os
import pickle
import subprocess
import sys

RESULTS = "results"


# --------------------------------------------------------------------------
# Finding the project folder
# --------------------------------------------------------------------------

def setup(repo=None, drive_folder="gqe-rna", verbose=True):
    """Mount Drive, find the project, and make results/ persistent.

    Works in three situations:
      1. Colab with the gqe-rna folder in My Drive
      2. Colab with gqe-rna.zip in My Drive (we unzip it once)
      3. Running locally, outside Colab

    Returns the project path.
    """
    if repo is not None and os.path.isdir(os.path.join(repo, "src")):
        _finish(repo, verbose)
        return repo

    repo = None

    # Case 1 and 2: we are on Colab.
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        root = "/content/drive/MyDrive"
        cand = os.path.join(root, drive_folder)

        if not os.path.isdir(os.path.join(cand, "src")):
            zp = os.path.join(root, drive_folder + ".zip")
            if os.path.exists(zp):
                if verbose:
                    print("unzipping", zp)
                subprocess.run(["unzip", "-q", "-o", zp, "-d", root], check=True)

        if os.path.isdir(os.path.join(cand, "src")):
            repo = cand
        else:
            raise FileNotFoundError(
                "Could not find the project in your Drive.\n"
                f"Put either the folder '{drive_folder}' or the file "
                f"'{drive_folder}.zip' at the top level of My Drive, "
                "then run this cell again.")
    except ImportError:
        # Case 3: not on Colab. Look nearby.
        for c in [".", "..", drive_folder, os.path.join("..", drive_folder)]:
            if os.path.isdir(os.path.join(c, "src")):
                repo = os.path.abspath(c)
                break
        if repo is None:
            raise FileNotFoundError("Could not find the src/ folder.")

    _finish(repo, verbose)
    return repo


def _finish(repo, verbose=True):
    """Put the project on the import path and make results/ persistent."""
    import importlib
    if repo not in sys.path:
        sys.path.insert(0, repo)
    importlib.invalidate_caches()
    os.chdir(repo)                       # so results/ is written into Drive
    os.makedirs(RESULTS, exist_ok=True)
    if verbose:
        print("project folder:", os.getcwd())


def install_vienna(verbose=True):
    """Install ViennaRNA if it is missing. Safe to run more than once."""
    try:
        import RNA                                        # noqa: F401
        if verbose:
            print("ViennaRNA already installed")
        return True
    except ImportError:
        pass
    if verbose:
        print("installing ViennaRNA ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ViennaRNA"],
                   check=False)
    try:
        import RNA                                        # noqa: F401
        return True
    except ImportError:
        print("ViennaRNA did not install. Try running this cell again, or\n"
              "Runtime -> Restart session, then run it once more.")
        return False


# --------------------------------------------------------------------------
# Passing results between notebooks
# --------------------------------------------------------------------------

def save(name, obj):
    """Save a result so the next notebook can pick it up."""
    path = os.path.join(RESULTS, name + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("saved", path)
    return path


def load(name):
    """Load a result saved by an earlier notebook."""
    with open(os.path.join(RESULTS, name + ".pkl"), "rb") as f:
        return pickle.load(f)


def require(name, made_by):
    """Load a result, with a clear message if the earlier notebook was skipped."""
    path = os.path.join(RESULTS, name + ".pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}.\n"
            f"Run notebook {made_by} first. Its results are saved to your "
            "Drive, so you only need to do it once.")
    return load(name)


def show_results():
    """List everything saved so far, and which notebook made it."""
    made_by = {"refs": "01", "encoding": "02", "gqe_run": "03",
               "gqe_model": "03", "baselines": "04"}
    print(f"{'file':<22}{'size':>10}   from notebook")
    print("-" * 50)
    if not os.path.isdir(RESULTS):
        print("(nothing yet)")
        return
    for f in sorted(os.listdir(RESULTS)):
        if f.startswith("."):
            continue
        kb = os.path.getsize(os.path.join(RESULTS, f)) / 1024
        stem = f.rsplit(".", 1)[0]
        print(f"{f:<22}{kb:9.1f}K   {made_by.get(stem, '?')}")


# --------------------------------------------------------------------------
# Getting files in and out of Colab
# --------------------------------------------------------------------------

def download_results(zip_name="gqe-rna-results.zip"):
    """Zip up results/ and download it to your computer."""
    subprocess.run(["zip", "-qr", zip_name, RESULTS], check=False)
    try:
        from google.colab import files
        files.download(zip_name)
        print("download started:", zip_name)
    except ImportError:
        print("saved", os.path.abspath(zip_name))
    return zip_name


def upload_into_results():
    """Upload files from your computer into results/. Rarely needed.

    Useful if a teammate ran a notebook and sent you their .pkl file.
    """
    try:
        from google.colab import files
    except ImportError:
        print("Only works on Colab. Copy the file into results/ by hand.")
        return []
    got = files.upload()
    moved = []
    for name in got:
        dest = os.path.join(RESULTS, name)
        os.replace(name, dest)
        moved.append(dest)
        print("moved to", dest)
    return moved


def save_figure(fig, name):
    """Save a plot into results/ so you can use it in your slides."""
    path = os.path.join(RESULTS, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("saved", path)
    return path
