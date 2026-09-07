"""Publish the Run A checkpoint to the Hugging Face Hub, gated on the evaluation.

This script will not upload anything unless ``reports/day2_runA.json`` -- the
output of ``scripts/eval_runA.py`` -- records a **PASSED** gate. That is the
whole design. Publishing weights is not reversible in any sense that matters:
they are downloaded, cached, and cited by people who will never see the run that
produced them. So the gate is enforced here, in code, and there is deliberately
no flag that overrides it. If the gate fails, fix the model.

The model card is **generated** from that same report by
``src.export.model_card.build_model_card``, not written by hand, so the
parameter count, the metric table, the training configuration and the caveats on
the card are the ones that were measured. In particular the parameter-budget
overage and the LPIPS caveat travel onto the card automatically; those are
exactly the two things that go missing when a card is typed from memory.

Credentials come from the environment (``cfg.huggingface.token_env``, default
``HF_TOKEN``) and never from a config file or an argument, so a token cannot be
committed or land in a shell history. The repository owner is read from the
token's own ``whoami``, which is why no username appears in version control.

    # No token and no network: generate the card, list what would be uploaded.
    python scripts/push_to_hf.py --dry-run

    # The real push (private repo by default; --public to publish openly).
    python scripts/push_to_hf.py

Exit codes: ``0`` uploaded (or dry run completed), ``2`` refused -- the gate
failed, the token is absent, or a file is missing -- ``1`` the run broke.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export.model_card import build_model_card  # noqa: E402
from src.utils.config import add_standard_args, load_config  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.paths import repo_root  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Upload the Run A checkpoint, its training log and a generated "
            "model card to the Hugging Face Hub. Refuses unless the evaluation "
            "report records a passed gate."
        ),
    )
    add_standard_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Generate the model card and print exactly what would be uploaded, "
            "then stop. Touches no network and needs no token."
        ),
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help=(
            "Create the repository public. Default is private "
            "(cfg.huggingface.private), because a repo can be made public in "
            "one click while weights someone has already downloaded cannot be "
            "unpublished."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Override cfg.eval_runA.report_json, the gated evaluation report.",
    )
    return parser.parse_args(argv)


def _resolve(path_like: Any) -> Path:
    """Resolve a config path against the repository root when it is relative."""
    path = Path(str(path_like))
    return path if path.is_absolute() else repo_root() / path


def project_url(logger: Any) -> str:
    """Read the project's source URL from the git remote.

    Resolved from ``git remote get-url origin`` rather than stored in config,
    for the same reason ``configs/kaggle_jobs.yaml`` carries no Kaggle username:
    an account name does not belong in version control, and a value derived from
    the repository is always the right one for whoever cloned it.

    Args:
        logger: Logger.

    Returns:
        An ``https://`` URL with any ``.git`` suffix and embedded credentials
        stripped, or ``""`` when there is no usable remote -- in which case the
        card names the project without linking it.
    """
    try:
        raw = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=str(repo_root()),
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning(
            "Could not read the git remote (%s), so the model card will name "
            "the project without linking it.", exc
        )
        return ""

    if raw.startswith("git@"):
        # git@host:owner/repo.git -> https://host/owner/repo
        host, _, path = raw[4:].partition(":")
        raw = f"https://{host}/{path}"
    if not raw.startswith("https://"):
        logger.warning(
            "Git remote %r is not an https URL; omitting the project link.", raw
        )
        return ""
    # A remote can carry a token (https://user:token@host/...). That must never
    # reach a published card.
    scheme, _, rest = raw.partition("://")
    if "@" in rest:
        rest = rest.rpartition("@")[2]
        logger.warning(
            "Stripped credentials from the git remote before putting it on the "
            "model card."
        )
    return f"{scheme}://{rest}".removesuffix(".git")


def load_gated_report(cfg: Any, override: Optional[str], logger: Any) -> Dict[str, Any]:
    """Read the evaluation report and refuse to continue unless it passed.

    Args:
        cfg: Loaded config; reads ``eval_runA.report_json``.
        override: ``--report`` value, or None.
        logger: Logger.

    Returns:
        The parsed report.

    Raises:
        FileNotFoundError: The report does not exist -- nothing has been
            evaluated, so there is nothing to publish.
        PermissionError: The report records a failed, or unevaluable, gate.
            Raised as a refusal rather than returned as a flag so that no later
            branch can proceed past it.
    """
    report_path = _resolve(
        override if override is not None else cfg["eval_runA"]["report_json"]
    )
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No evaluation report at {report_path}. Run "
            "'python scripts/eval_runA.py' first: the push is gated on its "
            "result and there is nothing to gate on yet."
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if bool(report.get("smoke", False)):
        raise PermissionError(
            f"{report_path} is a SMOKE report -- synthetic stub, random "
            "weights. Refusing to publish a model on the strength of it. Run "
            "'python scripts/eval_runA.py' without --smoke."
        )
    if report.get("max_batches") is not None:
        raise PermissionError(
            f"{report_path} was written by a TRUNCATED run "
            f"(--max-batches {report['max_batches']}), so its gate verdict "
            "describes a handful of batches rather than the validation split. "
            "Refusing to publish on it."
        )

    gate = report["gate"]
    if not gate["passed"]:
        failures = [
            f"{entry['metric']} ({entry['reason'] or 'did not improve'})"
            for entry in gate["metrics"]
            if not entry["passed"]
        ]
        raise PermissionError(
            "THE GATE FAILED, so nothing will be published. Unmet criteria: "
            + "; ".join(failures)
            + ". There is deliberately no flag to override this."
        )

    logger.info(
        "Gate PASSED in %s: %s.",
        report_path,
        ", ".join(
            f"{e['metric']} {e['delta_model_minus_bicubic']:+.4f}"
            for e in gate["metrics"]
        ),
    )
    return report


def resolve_token(cfg: Any) -> str:
    """Read the Hub write token from the environment.

    Args:
        cfg: Loaded config; reads ``huggingface.token_env``.

    Returns:
        The token.

    Raises:
        PermissionError: The variable is unset or empty, with the exact command
            to set it. Never falls back to a cached CLI login: the caller asked
            for this variable, and silently using a different credential is how
            an artefact lands in the wrong account.
    """
    name = str(cfg["huggingface"]["token_env"])
    token = os.environ.get(name, "").strip()
    if not token:
        raise PermissionError(
            f"${name} is not set, so there is no credential to publish with.\n\n"
            f"  PowerShell (this session):\n"
            f"      $env:{name} = \"hf_xxxxxxxxxxxxxxxxxxxx\"\n\n"
            f"  PowerShell (persist for future sessions):\n"
            f"      [Environment]::SetEnvironmentVariable("
            f"\"{name}\", \"hf_xxxxxxxxxxxxxxxxxxxx\", \"User\")\n\n"
            f"  bash:\n"
            f"      export {name}=hf_xxxxxxxxxxxxxxxxxxxx\n\n"
            "Create a token with WRITE scope at "
            "https://huggingface.co/settings/tokens"
        )
    return token


def collect_files(cfg: Any) -> List[Path]:
    """Locate the run artefacts to upload.

    Args:
        cfg: Loaded config; reads ``huggingface.files`` and
            ``eval_runA.run_dir``.

    Returns:
        Absolute paths, in the configured order.

    Raises:
        FileNotFoundError: Any listed file is missing. Checked together, up
            front, so a missing log.csv is not discovered after the 18 MB
            checkpoint has already been uploaded.
    """
    run_dir = _resolve(cfg["eval_runA"]["run_dir"])
    paths = [run_dir / str(name) for name in cfg["huggingface"]["files"]]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {[p.name for p in missing]} in {run_dir}. Fetch the "
            "Kaggle output with 'python scripts/kaggle_run.py fetch --job runa' "
            "and copy the run artefacts there."
        )
    return paths


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, smoke=args.smoke, overrides=args.overrides)

    logger = get_logger("push_to_hf", log_file=cfg.paths.log_file)
    seed_everything(cfg.seed)

    block = cfg["huggingface"]
    dry_run = bool(args.dry_run or args.smoke)

    try:
        report = load_gated_report(cfg, args.report, logger)
        files = collect_files(cfg)
    except (FileNotFoundError, PermissionError) as exc:
        logger.error("%s", exc)
        print(f"\nREFUSED.\n\n{exc}\n")
        return 2

    # -- the token, before anything expensive --------------------------------
    token = None
    if not dry_run:
        try:
            token = resolve_token(cfg)
        except PermissionError as exc:
            logger.error("No Hub token: %s", str(exc).splitlines()[0])
            print(f"\nREFUSED.\n\n{exc}\n")
            return 2

    # Imported here, not at module scope: --dry-run must work without the
    # package installed, and that is the mode someone reaches for first.
    if not dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            message = (
                "huggingface_hub is not installed in this interpreter.\n\n"
                "      D:\\SIH\\DrishtiSR\\.venv\\Scripts\\python.exe -m pip "
                "install huggingface_hub\n\n"
                "Use that exact interpreter -- see the Python-versions table in "
                "AGENTS.md; a bare 'pip' here installs into a different Python."
            )
            logger.error("%s", message)
            print(f"\nREFUSED.\n\n{message}\n")
            raise SystemExit(2) from exc

        api = HfApi(token=token)
        owner = api.whoami()["name"]
        repo_id = f"{owner}/{block['repo_name']}"
    else:
        # The owner is only knowable from the token. Say so rather than
        # inventing a plausible-looking username in the dry run's output.
        repo_id = f"<your-hf-username>/{block['repo_name']}"

    # -- the model card, generated from the report ---------------------------
    card = build_model_card(report, cfg, repo_id, project_url(logger))
    card_path = _resolve(block["card_output"])
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(card, encoding="utf-8")
    logger.info("Model card written to %s (%d lines)", card_path,
                len(card.splitlines()))

    private = bool(block["private"]) and not args.public
    total_mb = sum(path.stat().st_size for path in files) / 1024 / 1024

    print()
    print(f"  repo:     {repo_id}  ({'private' if private else 'PUBLIC'})")
    print(f"  card:     {card_path}  -> {block['card_name']}")
    for path in files:
        print(f"  upload:   {path.name:<12} {path.stat().st_size / 1024:>10.1f} KB")
    print(f"  total:    {total_mb:.2f} MB plus the card")
    print()

    if dry_run:
        print(
            "DRY RUN -- nothing was uploaded. The card above was written "
            "locally so it can be read before it is published. Re-run without "
            "--dry-run to push."
        )
        return 0

    api.create_repo(
        repo_id=repo_id,
        repo_type=str(block["repo_type"]),
        private=private,
        exist_ok=True,
    )
    logger.info("Repository %s ready (private=%s).", repo_id, private)

    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo=str(block["card_name"]),
        repo_id=repo_id,
        repo_type=str(block["repo_type"]),
        commit_message=str(block["commit_message"]).strip(),
    )
    for path in files:
        logger.info("Uploading %s (%.2f MB)...", path.name,
                    path.stat().st_size / 1024 / 1024)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type=str(block["repo_type"]),
            commit_message=str(block["commit_message"]).strip(),
        )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Published to %s", url)
    print(f"\nPublished: {url}")
    if private:
        print(
            "The repository is PRIVATE. Make it public from Settings on the "
            "Hub, or re-run with --public."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
