"""Guarded NN lifecycle Package 01 installer. Standard library only."""

import argparse
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def safe(root, name):
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError("Unsafe path: " + name)
    if any(p.is_symlink() for p in path.parents if p.is_relative_to(root)):
        raise ValueError("Symlink parent: " + name)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path.cwd())
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--rollback", action="store_true")
    p.add_argument("--emit-patch", type=Path)
    args = p.parse_args()
    root = args.repo.resolve()
    delivery = Path(__file__).resolve().parent
    backup = safe(root, ".nn_lifecycle_01_backup")
    if args.rollback:
        state = json.loads((backup / "installation.json").read_text())
        for name, item in state.items():
            target = safe(root, name)
            if not target.is_file() or sha(target.read_bytes()) != item["after"]:
                raise ValueError("Later edits prevent rollback: " + name)
            if (
                item["before"] is not None
                and sha(safe(backup / "originals", name).read_bytes()) != item["before"]
            ):
                raise ValueError("Original backup differs: " + name)
        for name, item in state.items():
            if item["before"] is None:
                (root / name).unlink()
            else:
                shutil.copy2(backup / "originals" / name, root / name)
        print("ROLLED BACK; backup retained for audit")
        return
    if backup.exists():
        state = json.loads((backup / "installation.json").read_text())
        if all(
            safe(root, n).is_file() and sha((root / n).read_bytes()) == v["after"]
            for n, v in state.items()
        ):
            print("Already installed; no changes")
            return
        raise ValueError("Existing backup with changed/rolled-back files; review before retrying")
    spec = json.loads((delivery / "manifest.json").read_text())
    for name, expected in spec["dependencies"].items():
        actual = blob(safe(root, name).read_bytes())
        if actual != expected:
            raise ValueError(f"Dependency differs: {name}; expected={expected}, actual={actual}")
    prepared = {}
    for name, item in spec["edits"].items():
        before = safe(root, name).read_bytes()
        actual = blob(before)
        if actual != item["git_blob"]:
            raise ValueError(
                f"Source differs: {name}; expected={item['git_blob']}, actual={actual}"
            )
        after = before.decode("utf-8")
        for change in item["changes"]:
            if after.count(change["old"]) != 1:
                raise ValueError("Missing/ambiguous source anchor: " + name)
            after = after.replace(change["old"], change["new"])
        prepared[name] = (before, after.encode("utf-8"))
    for name, expected in spec["new_files"].items():
        target = safe(root, name)
        if target.exists():
            raise FileExistsError("New destination exists: " + name)
        data = safe(delivery / "payload", name).read_bytes()
        if sha(data) != expected:
            raise ValueError("Payload hash differs: " + name)
        prepared[name] = (None, data)
    for name, (_, data) in prepared.items():
        if name.endswith(".py"):
            ast.parse(data.decode(), filename=name, feature_version=(3, 11))
            compile(data.decode(), name, "exec")
    patch = ""
    for name, (before, after) in prepared.items():
        patch += "diff --git a/" + name + " b/" + name + "\n"
        if before is None:
            patch += "new file mode 100644\n"
        patch += "".join(
            difflib.unified_diff(
                [] if before is None else before.decode().splitlines(True),
                after.decode().splitlines(True),
                fromfile="/dev/null" if before is None else "a/" + name,
                tofile="b/" + name,
            )
        )
    if args.emit_patch:
        with args.emit_patch.open("x", encoding="utf-8") as stream:
            stream.write(patch)
    if args.check:
        print(
            "CHECK OK:",
            len(prepared),
            "files; byte-hash guards, unique anchors, payload hashes and syntax",
        )
        return
    backup.mkdir(exist_ok=False)
    state = {
        name: {"before": None if before is None else sha(before), "after": sha(after)}
        for name, (before, after) in prepared.items()
    }
    for name, (before, _) in prepared.items():
        if before is not None:
            dest = backup / "originals" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, dest)
    (backup / "installation.json").write_text(json.dumps(state, indent=2))
    (backup / "applied.patch").write_text(patch)
    written = []
    try:
        for name, (before, after) in prepared.items():
            target = safe(root, name)
            if before is None:
                if target.exists():
                    raise ValueError("Destination appeared during installation: " + name)
            elif target.read_bytes() != before:
                raise ValueError("Source changed during installation: " + name)
            target.parent.mkdir(parents=True, exist_ok=True)
            written.append(name)
            target.write_bytes(after)
    except Exception:
        for name in reversed(written):
            if prepared[name][0] is None:
                (root / name).unlink(missing_ok=True)
            else:
                shutil.copy2(backup / "originals" / name, root / name)
        raise
    print("APPLIED:", len(prepared), "files; backups and exact diff in", backup)


if __name__ == "__main__":
    main()
