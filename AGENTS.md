# Publishing rules for this branch (reports.eric-heiden.com)

This branch is PUBLIC and served live. Everything committed here is world-readable the moment it is pushed. Any agent working in this branch MUST follow these rules.

## What may be published
- Static report pages only: HTML files and the presentation assets they reference (CSS, JS libraries, images, video, audio).
- Links may only point to public repositories and public websites.

## What must NEVER be committed or pushed
- No code: no source files (.py, .sh, notebooks), no tools/, scripts/, or build-output directories.
- No data files: no .json/.jsonl/.csv/.npz/checkpoint files, no data/ directories.
- No logs, transcripts, or agent artifacts: no .log/.stderr files, no agent session transcripts, no archive bundles (.zip/.tar.gz) of work directories.
- No internal information in ANY file, including inside HTML pages:
  - no local filesystem paths (home directories, worktree paths, /home/..., /Users/..., C:\...)
  - no machine or host names
  - no internal network names or addresses (Tailscale / .ts.net names, tailnet or LAN IPs)
  - no internal URLs, dashboards, or service endpoints
  - no usernames or credentials of any kind

## Before every push
1. Review `git status`: commit only the files you intend to publish. Never `git add -A` a raw work directory.
2. Scan everything being added for internal strings: local paths, host names, .ts.net, private IPs. If anything matches, stop and clean it first.
3. The .gitignore in this branch blocks the known leak classes. Do not force-add (-f) around it.

If a report was generated alongside code, data, or transcripts, publish ONLY the rendered static files, after the scan.
