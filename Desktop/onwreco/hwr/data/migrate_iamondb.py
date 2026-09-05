import argparse
import os
import shutil

from hwr.constants import PATH


REQUIRED_ITEMS = ("lineStrokes(on)", "split-config")


def _resolve_item_path(source_root, item):
    direct = os.path.join(source_root, item)
    if os.path.exists(direct):
        return direct
    nested = os.path.join(source_root, "data", "iamon", item)
    if os.path.exists(nested):
        return nested
    return None


def _remove_if_exists(path):
    if os.path.islink(path) or os.path.isfile(path):
        os.unlink(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def migrate(source_root, target_root=PATH.DATA_DIR, copy=False, force=False):
    source_root = os.path.abspath(source_root)
    target_root = os.path.abspath(target_root)
    os.makedirs(target_root, exist_ok=True)

    resolved_paths = {}
    for item in REQUIRED_ITEMS:
        source_item = _resolve_item_path(source_root, item)
        if source_item is None:
            raise FileNotFoundError(
                f"Could not find '{item}' in '{source_root}'."
            )
        resolved_paths[item] = source_item

    for item, source_item in resolved_paths.items():
        target_item = os.path.join(target_root, item)
        if os.path.exists(target_item) or os.path.islink(target_item):
            if not force:
                raise FileExistsError(
                    f"Target already exists: {target_item}. Use --force to replace it."
                )
            _remove_if_exists(target_item)

        if copy:
            shutil.copytree(source_item, target_item)
        else:
            os.symlink(source_item, target_item, target_is_directory=True)

    return resolved_paths


def main():
    parser = argparse.ArgumentParser(
        description="Migrate IAMOnDB files to the project's expected layout."
    )
    parser.add_argument("source", help="Path to IAMOnDB root directory")
    parser.add_argument(
        "--target",
        default=PATH.DATA_DIR,
        help="Target data directory (defaults to PATH.DATA_DIR)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target entries",
    )
    args = parser.parse_args()

    moved = migrate(
        source_root=args.source,
        target_root=args.target,
        copy=args.copy,
        force=args.force,
    )
    print("Migration completed.")
    for key, value in moved.items():
        print(f"{key}: {value}")
    print("\nNext step (optional retraining prep):")
    print("python -m hwr.data.createnpz 6")


if __name__ == "__main__":
    main()
