"""Verify the frozen public release and, optionally, its local study inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
TEXT_NORMALIZED = {".csv", ".json", ".md", ".py", ".yml"}


def file_data(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"File outside the declared root: {relative}")
    data = path.read_bytes()
    if path.suffix.lower() in TEXT_NORMALIZED or path.name in {".gitattributes", ".gitignore"}:
        return data.replace(b"\r\n", b"\n")
    return data


def check_files(root, records):
    for entry in records:
        actual = sha256(file_data(root, entry["path"])).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {entry['path']}")


def collection_digest(root, folder, pattern):
    path = (root / folder).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"Collection outside the declared root: {folder}")
    files = sorted(p for p in path.glob(pattern) if p.is_file())
    # Include each relative name in the local digest, and store only the collection digest.
    digest = sha256()
    for p in files:
        rel = p.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8") + b"\0")
        digest.update(sha256(file_data(path, rel)).digest())
    return len(files), digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Also check the authorized historical worktree and archive inputs")
    options = parser.parse_args()
    manifest = json.loads((HERE / "workflow/Freeze_Manifest.json").read_text(encoding="utf-8"))
    check_files(HERE, manifest["public_files"])
    print(f"Verified {len(manifest['public_files'])} public files against the frozen SHA-256 manifest.")
    if options.workspace is None:
        return
    root = options.workspace.resolve(strict=True)
    check_files(root, manifest["local_dependencies"])
    check_files(root, manifest["local_archive_files"])
    for row in manifest["local_archive_collections"]:
        count, digest = collection_digest(root, row["path"], row["glob"])
        if (count, digest) != (row["count"], row["sha256"]):
            raise ValueError(f"Frozen archive collection differs: {row['role']}")
    print("Verified the local implementation, frozen scan manifest, label archive, and synthetic cohort diagnostics.")


if __name__ == "__main__":
    main()
