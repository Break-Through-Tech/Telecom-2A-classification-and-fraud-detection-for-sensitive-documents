#!/usr/bin/env python3
"""Commit pending Git changes in small batches; optionally push each batch.

Run from your repository (or pass --repo PATH). No third-party packages needed.

  python batch_commit.py --dry-run
  python batch_commit.py --batch-size 100
  python batch_commit.py --batch-size 100 --push

Skips the paths in EXCLUDED_PATHS, including already-staged changes.
Includes new, modified, deleted, and already-staged files, honoring Git ignores
for untracked files. Each selected path includes all its current edits, including
any edits that were previously only partially staged. Existing commits are not
rewritten. Do not edit files or run other Git operations during the run.
"""
import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time


# Repository-relative paths. Git reports forward slashes on Windows too.
EXCLUDED_PATHS = (
    b'data/files-archive',
    b'data/files-archive.zip',
    b'idnet_features.csv',
    b'idnet_processed_manifest.csv',
    b'idnet_features_threshold_audit.csv',
    b'idnet_features_top150_selected_features.csv',
    b'idnet_features_top150_threshold_audit.csv',
    b'sample_manifest.csv',
    b'generated_manifest.csv',
    b'pipeline-backup',
    b'runs',
)


def excluded(path):
    return any(path == prefix or path.startswith(prefix + b'/') for prefix in EXCLUDED_PATHS)


class GitError(RuntimeError):
    pass


class Repository:
    def __init__(self, directory):
        self.env = os.environ.copy()
        self.env['GIT_LITERAL_PATHSPECS'] = '1'
        self.directory = Path(directory).resolve()
        root = self.git('rev-parse', '--show-toplevel', capture=True).stdout
        self.directory = Path(os.fsdecode(root.rstrip(b'\r\n')))

    def git(self, *args, capture=False, data=None, check=True):
        result = subprocess.run(
            ['git', *args], cwd=self.directory, env=self.env,
            input=data, stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
        if check and result.returncode:
            detail = (result.stderr or b'').decode('utf-8', errors='replace').strip()
            raise GitError(f"git {args[0]} failed (exit {result.returncode}). {detail}")
        return result

    def head(self):
        result = self.git('rev-parse', '--verify', 'HEAD', capture=True, check=False)
        return result.stdout.strip().decode('ascii') if result.returncode == 0 else None

    def check_state(self):
        for marker in ('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply'):
            location = self.git('rev-parse', '--git-path', marker, capture=True).stdout
            path = Path(os.fsdecode(location.rstrip(b'\r\n')))
            if not path.is_absolute():
                path = self.directory / path
            if path.exists():
                raise GitError('Finish or abort the active merge/rebase/cherry-pick/revert first.')

    def changes(self):
        # Disable rename pairing so each NUL-delimited record has exactly one
        # path. A rename is handled as a deletion and addition, possibly in
        # different batches. No command-line filename length limits.
        output = self.git(
            '-c', 'status.renames=false', 'status', '--porcelain=v1',
            '-z', '--untracked-files=all', '--ignore-submodules=none',
            capture=True,
        ).stdout
        paths = []
        for record in output.split(b'\0'):
            if not record:
                continue
            status, path = record[:2], record[3:]
            if excluded(path):
                continue
            if status in (b'DD', b'AU', b'UD', b'UA', b'DU', b'AA', b'UU'):
                raise GitError('Resolve merge conflicts before batching.')
            absolute = self.directory / os.fsdecode(path)
            if absolute.is_dir():
                raise GitError(f'Changed directory/submodule needs separate handling: {os.fsdecode(path)}')
            paths.append(path)
        return sorted(set(paths))

    def size(self, path):
        try:
            return (self.directory / os.fsdecode(path)).lstat().st_size
        except FileNotFoundError:
            return 0  # Deleted files need to be committed too.

    def push(self, commit, remote, branch, attempts, delay):
        for attempt in range(1, attempts + 1):
            print(f'Pushing {commit[:12]} to {remote}/{branch} ({attempt}/{attempts})...', flush=True)
            result = self.git(
                'push', remote, f'{commit}:refs/heads/{branch}', check=False,
            )
            if result.returncode == 0:
                return
            if attempt < attempts:
                time.sleep(min(delay * attempt, 30))
        raise GitError(
            'Push failed after retries. All completed commits remain local. '
            'Fix the connection/authentication or remote rejection, then rerun '
            'the same command. No history has been rewritten or force-pushed.'
        )


def plan_batches(repo, paths, limit, byte_limit):
    batches, current, size = [], [], 0
    for path in paths:
        file_size = repo.size(path)
        if current and (len(current) >= limit or size + file_size > byte_limit):
            batches.append(current)
            current, size = [], 0
        current.append(path)
        size += file_size
    if current:
        batches.append(current)
    return batches


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be at least 1')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default='.', help='Repository folder; default: current folder')
    parser.add_argument('--batch-size', type=positive_int, default=100, help='Maximum paths per commit (default: 100)')
    parser.add_argument('--max-mb', type=float, default=25, help='Approximate raw file MB per batch (default: 25; a larger single file gets its own batch)')
    parser.add_argument('--message', default='Add project files', help='Commit message prefix')
    parser.add_argument('--push', action='store_true', help='Push current HEAD first, then every new commit, with retries')
    parser.add_argument('--remote', default='origin', help='Push remote (default: origin)')
    parser.add_argument('--branch', help='Remote target branch (default: current local branch)')
    parser.add_argument('--retries', type=positive_int, default=4, help='Push attempts (default: 4)')
    parser.add_argument('--retry-delay', type=positive_int, default=5, help='Initial retry delay in seconds (default: 5; capped at 30)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without staging, committing, or pushing')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_mb) or args.max_mb <= 0:
        parser.error('--max-mb must be a positive finite number')
    repo = Repository(args.repo)
    repo.check_state()
    paths = repo.changes()
    batches = plan_batches(repo, paths, args.batch_size, args.max_mb * 1024 * 1024)
    print(f'Repository: {repo.directory}', flush=True)
    print('Skipping: ' + ', '.join(os.fsdecode(p) for p in EXCLUDED_PATHS), flush=True)
    print(f'{len(paths)} changed paths; {len(batches)} batches (at most {args.batch_size} paths / about {args.max_mb:g} MB each).', flush=True)
    if args.dry_run:
        for i, batch in enumerate(batches, 1):
            mb = sum(repo.size(p) for p in batch) / (1024 * 1024)
            examples = ', '.join(repr(os.fsdecode(p)) for p in batch[:3])
            print(f'Batch {i}: {len(batch)} paths, {mb:.2f} MB; {examples}', flush=True)
        if args.push:
            print('Real run will push existing HEAD first, then each batch.')
        return 0
    if batches:
        # Fail before staging if Git does not have a commit identity.
        repo.git('var', 'GIT_AUTHOR_IDENT', capture=True)
        repo.git('var', 'GIT_COMMITTER_IDENT', capture=True)
    branch = args.branch
    if args.push:
        if not branch:
            result = repo.git('symbolic-ref', '--quiet', '--short', 'HEAD', capture=True, check=False)
            if result.returncode:
                raise GitError('Detached HEAD: check out a branch before pushing.')
            branch = result.stdout.strip().decode('utf-8')
        repo.git('check-ref-format', 'refs/heads/' + branch, capture=True)
        # Prevent accumulated unpushed commits from being bundled with the
        # first new batch. Also completes an interrupted push on a rerun.
        existing = repo.head()
        if existing:
            repo.push(existing, args.remote, branch, args.retries, args.retry_delay)
    for i, batch in enumerate(batches, 1):
        print(f'\nCommitting batch {i}/{len(batches)} ({len(batch)} paths)...', flush=True)
        data = b'\0'.join(batch) + b'\0'
        tracked = set(repo.git('ls-files', '--cached', '-z', capture=True).stdout.split(b'\0'))
        # An already-staged deletion is absent from both disk and the index.
        # Re-adding that missing path would fail; commit --only can still use it.
        to_add = [p for p in batch if p in tracked or os.path.lexists(repo.directory / os.fsdecode(p))]
        if to_add:
            add_data = b'\0'.join(to_add) + b'\0'
            repo.git('add', '-A', '--pathspec-from-file=-', '--pathspec-file-nul', data=add_data)
        staged = set(repo.git('diff', '--cached', '--name-only', '--no-renames', '-z', capture=True).stdout.split(b'\0'))
        batch = [p for p in batch if p in staged]
        if not batch:
            print('No net changes in this batch; skipping.')
            continue
        data = b'\0'.join(batch) + b'\0'
        # --only prevents other already-staged paths leaking into this batch.
        repo.git('commit', '--only', '-m', f'{args.message} (batch {i}/{len(batches)})',
                 '--pathspec-from-file=-', '--pathspec-file-nul', data=data)
        if args.push:
            repo.push(repo.head(), args.remote, branch, args.retries, args.retry_delay)
    remaining = repo.changes()
    if remaining:
        print(f'{len(remaining)} paths changed during the run or via hooks; rerun to include them.')
    else:
        print('Done. All included pending changes committed' + (' and pushed.' if args.push else '.'))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped. Completed commits remain local; rerun to continue.', file=sys.stderr)
        sys.exit(130)
    except (GitError, OSError) as error:
        print(f'\nStopped: {error}', file=sys.stderr)
        sys.exit(1)
