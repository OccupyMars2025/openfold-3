import importlib
import os
from pathlib import Path


def prep_deepspeed():
    # deepspeed requires the envvar set, but doesn't care about value
    os.environ["CUTLASS_PATH"] = os.environ.get("CUTLASS_PATH", "placeholder")


def prep_cutlass():
    cutlass_lib_is_installed = importlib.util.find_spec("cutlass_library") is not None
    cutlass_path = Path(os.environ.get("CUTLASS_PATH", "placeholder"))

    # This workaround is used when the conda environment is created with the
    # environments/production.yml + installation of cutlass repo
    if not cutlass_lib_is_installed:
        if not cutlass_path.exists():
            raise OSError(
                "CUTLASS_PATH environment variable is not set to a valid path, "
                "and cutlass_library is not installed. Please install nvidia-cutlass"
                "via pip or set CUTLASS_PATH to the root of a local cutlass clone."
            )

        return

    # otherwise, apparently need to set the headers for cutlass
    import cutlass_library

    headers_dir = Path(cutlass_library.__file__).parent / "source/include"
    cpath = os.environ.get("CPATH", "")
    # TODO: technically, this test should be a little fancier
    if str(headers_dir.resolve()) not in cpath:
        if cpath:
            cpath += ":"

        os.environ["CPATH"] = cpath + str(headers_dir.resolve())
