from pathlib import Path
import shutil

import openpi.models_pytorch.transformers_replace
import transformers


def main():
    print("Installing OpenPI overrides for transformers...")

    transformers_replace_dir = Path(openpi.models_pytorch.transformers_replace.__path__[0])
    transformers_root = Path(transformers.__path__[0])
    print(f"Copying from {transformers_replace_dir} to {transformers_root}")

    for item in transformers_replace_dir.iterdir():
        target = transformers_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)

    print("Done")


if __name__ == "__main__":
    main()
