"""Shared company-wide storage roots and safe legacy Sora migration."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from src.storage_paths import sora_artifacts,sora_releases,sora_run
from berhot_storage.paths import (
    repository_root,storage_root,artifact_root,release_root,new_artifact_run,
)
from scripts.migrate_storage import migrate


class SharedStorageTests(unittest.TestCase):
    def test_canonical_root_independent_of_current_working_directory(self):
        root=repository_root()
        self.assertTrue((root/"models"/"sora").is_dir())
        self.assertTrue((root/"docs"/"storage.md").is_file())
        self.assertEqual(storage_root(),root/"storage")
        self.assertEqual(
            sora_artifacts(),root/"storage"/"artifacts"/"sora"/"nano"/"mvp"
        )
        self.assertEqual(
            sora_releases(),root/"storage"/"releases"/"sora"/"nano"
        )
        oldcwd=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(
                    sora_artifacts(),root/"storage"/"artifacts"/"sora"/"nano"/"mvp"
                )
                generated=sora_run("pixel")
                self.assertEqual(generated.parent,sora_artifacts())
                self.assertFalse(generated.exists())
            finally:
                os.chdir(oldcwd)

    def test_common_helper_supports_all_families_and_unique_runs(self):
        for family in ("jsm","nabra","sora","fann","bahr","edrak"):
            directory=artifact_root(family,"nano","mvp")
            self.assertEqual(directory.parts[-4:],
                             ("artifacts",family,"nano","mvp"))
            self.assertEqual(release_root(family,"nano").parts[-3:],
                             ("releases",family,"nano"))
            self.assertEqual(new_artifact_run(family,"nano","mvp","train").parent,
                             directory)
        first=sora_run("pixel")
        second=sora_run("pixel")
        self.assertNotEqual(first,second)
        self.assertEqual(
            new_artifact_run("sora","nano","mvp","pixel",
                             override_root="/tmp/custom-output").parent,
            Path("/tmp/custom-output")
        )
        self.assertEqual(release_root("sora","nano","0.1.0").name,"0.1.0")
        for part in ("../nabra","/tmp","",".","a/b"):
            with self.assertRaises(ValueError):
                artifact_root(part,"nano")
        with self.assertRaises(ValueError):
            release_root("sora","nano","../../../bad")
        with self.assertRaises(ValueError):
            sora_run("../bad")

    def test_migration_dry_run_is_non_destructive_and_apply_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/"old"/"mvp"
            dst=root/"storage"/"artifacts"/"sora"/"nano"/"mvp"
            model=src/"pixel-20261103T010000Z-deadbeef"
            model.mkdir(parents=True)
            (model/"checkpoint.pt").write_bytes(b"fake weights fixture")
            (model/"report.json").write_text('{"ok":true}\n')
            before=hashlib.sha256((model/"checkpoint.pt").read_bytes()).hexdigest()
            plan=migrate(src,dst)
            self.assertEqual(plan["mode"],"dry-run")
            self.assertEqual(plan["runs"][0]["status"],"would-copy")
            self.assertFalse(dst.exists())
            result=migrate(src,dst,apply=True)
            self.assertEqual(result["runs"][0]["status"],"copied-verified")
            copied=dst/model.name/"checkpoint.pt"
            self.assertTrue(copied.is_file())
            self.assertEqual(
                hashlib.sha256(copied.read_bytes()).hexdigest(),before
            )
            self.assertEqual(
                hashlib.sha256((model/"checkpoint.pt").read_bytes()).hexdigest(),
                before
            )
            again=migrate(src,dst,apply=True)
            self.assertEqual(again["runs"][0]["status"],"already-identical")
            copied.write_bytes(b"collision")
            with self.assertRaises(FileExistsError):
                migrate(src,dst,apply=True)
            self.assertEqual((model/"checkpoint.pt").read_bytes(),b"fake weights fixture")

    def test_selective_run_migration_limits_extra_disk_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/"legacy"
            dst=root/"central"
            one=src/"pixel-important"
            two=src/"pixel-failed"
            one.mkdir(parents=True)
            two.mkdir(parents=True)
            (one/"checkpoint.pt").write_bytes(b"best")
            (two/"checkpoint.pt").write_bytes(b"unwanted")
            plan=migrate(src,dst,run_ids=["pixel-important"])
            self.assertEqual([r["run"] for r in plan["runs"]],
                             ["pixel-important"])
            result=migrate(src,dst,apply=True,run_ids=["pixel-important"])
            self.assertEqual(result["runs"][0]["status"],"copied-verified")
            self.assertEqual((dst/"pixel-important"/"checkpoint.pt").read_bytes(),
                             b"best")
            self.assertFalse((dst/"pixel-failed").exists())
            self.assertEqual((two/"checkpoint.pt").read_bytes(),b"unwanted")
            for unsafe in ("../unsafe","/tmp/unsafe",".",""):
                with self.assertRaises(ValueError):
                    migrate(src,dst,run_ids=[unsafe])

    def test_migration_guards_nested_destinations_and_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            src=root/"legacy"
            src.mkdir()
            with self.assertRaises(ValueError):
                migrate(src,src)
            with self.assertRaises(ValueError):
                migrate(src,src/"new")
            with self.assertRaises(ValueError):
                migrate(src,root)
            absent=migrate(root/"missing",root/"elsewhere")
            self.assertEqual(absent["status"],"legacy source not present")
            self.assertFalse((root/"elsewhere").exists())
            run=src/"spatial-trial"
            run.mkdir()
            outside=root/"checkpoint.pt"
            outside.write_bytes(b"external")
            (run/"checkpoint.pt").symlink_to(outside)
            with self.assertRaises(ValueError):
                migrate(src,root/"central")
            self.assertEqual(outside.read_bytes(),b"external")
