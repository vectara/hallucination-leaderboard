"""Prepare (and optionally open) the leaderboard PR to `main` from lb-engine results.

main's README table is the public record and has hand-curated display names and its own
tie order, so it is never regenerated. Instead this script:

1. reads main's current table (origin/main) and lb-engine's generated table (./README.md,
   after `update_readme.py`),
2. maps lb-engine names to main's display names via ALIASES,
3. keeps every existing main row exactly as it is and inserts only models main does not
   have yet, each after the last row with an equal or lower hallucination rate,
4. refuses to continue if an existing model's numbers differ, a model would disappear,
   or the result is not sorted,
5. updates the "Last updated" line, and the plot line + img/ file only if the top 25
   (answer rate > 95 %, as in update_readme.py) changed,
6. commits on a new branch in a git worktree from origin/main.

Nothing but README.md and img/ ever goes to main. The PR is merged by a reviewer.

Usage (from the repo root, after compile_results + update_readme.py):
    ../lb_env/bin/python main_readme_pr.py --worktree <scratch>/pr-main           # prepare + show diff
    ../lb_env/bin/python main_readme_pr.py --worktree <scratch>/pr-main --open-pr # push + open PR
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import date

# lb-engine name -> name shown on main. Add a line when a rename is requested.
ALIASES = {
    "mistralai/ministral-3b-2512": "mistralai/ministral-3-3b-2512",
    "mistralai/ministral-8b-2512": "mistralai/ministral-3-8b-2512",
    "mistralai/ministral-14b-2512": "mistralai/ministral-3-14b-2512",
    "mistralai/mistral-large-2512": "mistralai/mistral-3-large-2512",
    "zai-org/glm-4p7-flash": "zai-org/GLM-4.7-flash",
}

ROW_RE = re.compile(r"^\|([^|]+)\|([\d.]+) %\|([\d.]+) %\|([\d.]+) %\|([\d.]+)\|$")
PLOT_RE = re.compile(r"^!\[Plot: hallucination rates of various LLMs\]\(\./(img/[^)]+)\)$", re.M)
DATE_RE = re.compile(r"^Last updated on .+$", re.M)


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def parse_rows(readme):
    rows = []
    for line in readme.splitlines():
        m = ROW_RE.match(line)
        if m:
            rows.append({"name": m.group(1), "rate": float(m.group(2)), "answer": float(m.group(4)), "line": line})
    return rows


def top25(rows):
    return [r["name"] for r in rows if r["answer"] > 95][:25]


def fail(msg):
    sys.exit("ABORT: " + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worktree", required=True, help="where to create the main worktree (scratch dir)")
    ap.add_argument("--date", default=date.today().isoformat(), help="update date YYYY-MM-DD (default today)")
    ap.add_argument("--open-pr", action="store_true", help="push the prepared branch and open the PR")
    args = ap.parse_args()

    d = date.fromisoformat(args.date)
    branch = "lb-update-%d-%d-%d" % (d.month, d.day, d.year)

    if args.open_pr:
        if not os.path.isdir(args.worktree):
            fail("no prepared worktree at %s; run without --open-pr first" % args.worktree)
        title = git("log", "-1", "--format=%s", cwd=args.worktree).strip()
        body = git("log", "-1", "--format=%b", cwd=args.worktree).split("Co-Authored-By")[0].strip()
        git("push", "-u", "origin", branch, cwd=args.worktree)
        out = subprocess.run(["gh", "pr", "create", "--base", "main", "--head", branch, "--title", title,
                              "--body", body + "\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"],
                             cwd=args.worktree, check=True, capture_output=True, text=True).stdout
        print(out.strip())
        return

    git("fetch", "-q", "origin", "main")
    main_readme = git("show", "origin/main:README.md")
    lb_readme = open("README.md").read()

    main_rows = parse_rows(main_readme)
    lb_rows = parse_rows(lb_readme)
    for r in lb_rows:
        alias = ALIASES.get(r["name"])
        if alias:
            r["line"] = r["line"].replace("|%s|" % r["name"], "|%s|" % alias, 1)
            r["name"] = alias

    main_by = {r["name"]: r for r in main_rows}
    lb_by = {r["name"]: r for r in lb_rows}
    missing = sorted(set(main_by) - set(lb_by))
    if missing:
        fail("models on main but not in lb-engine table (alias missing?): %s" % missing)
    changed = [n for n in main_by if main_by[n]["line"] != lb_by[n]["line"]]
    if changed:
        fail("existing rows differ between main and lb-engine: %s" % changed)
    new = [r for r in lb_rows if r["name"] not in main_by]
    if not new:
        fail("no new models to add")

    rows = list(main_rows)
    for r in sorted(new, key=lambda r: r["rate"]):
        pos = max((i + 1 for i, x in enumerate(rows) if x["rate"] <= r["rate"]), default=0)
        rows.insert(pos, r)
    rates = [r["rate"] for r in rows]
    if any(b < a for a, b in zip(rates, rates[1:])):
        fail("resulting table is not sorted")
    if len(rows) != len(main_rows) + len(new):
        fail("row count mismatch")

    # Rebuild README: replace the table body lines in place, keep everything else as on main.
    lines = main_readme.splitlines(keepends=True)
    idx = [i for i, l in enumerate(lines) if ROW_RE.match(l.rstrip("\n"))]
    if idx != list(range(idx[0], idx[-1] + 1)):
        fail("main's table rows are not contiguous")
    out = lines[:idx[0]] + [r["line"] + "\n" for r in rows] + lines[idx[-1] + 1:]
    readme = "".join(out)
    readme = DATE_RE.sub("Last updated on %s %d, %d" % (d.strftime("%B"), d.day, d.year), readme, count=1)

    plot = None
    if top25(rows) != top25(main_rows):
        m = PLOT_RE.search(lb_readme)
        if not m or not os.path.isfile(m.group(1)):
            fail("top 25 changed but lb-engine README has no existing plot file")
        plot = m.group(1)
        readme = PLOT_RE.sub("![Plot: hallucination rates of various LLMs](./%s)" % plot, readme, count=1)

    if os.path.exists(args.worktree):
        fail("worktree path already exists: %s" % args.worktree)
    if git("ls-remote", "--heads", "origin", branch).strip():
        fail("branch %s already exists on origin" % branch)
    git("worktree", "add", "-q", "-b", branch, args.worktree, "origin/main")
    open(os.path.join(args.worktree, "README.md"), "w").write(readme)
    to_add = ["README.md"]
    if plot:
        shutil.copy(plot, os.path.join(args.worktree, plot))
        to_add.append(plot)
    git("add", *to_add, cwd=args.worktree)

    names = ", ".join(r["name"].split("/")[-1] for r in sorted(new, key=lambda r: r["rate"]))
    title = "Leaderboard update %s: %s" % (d.isoformat(), names)
    body = "\n".join("Adds %s: **%.1f %%** hallucination rate." % (r["name"], r["rate"])
                     for r in sorted(new, key=lambda r: r["rate"]))
    if plot:
        body += "\n\nTop 25 changed, so the plot is updated."
    git("commit", "-q", "-m", title + "\n\n" + body + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
        cwd=args.worktree)

    print("Prepared branch %s in %s (not pushed)\n" % (branch, args.worktree))
    for r in sorted(new, key=lambda r: r["rate"]):
        print("  new: %-45s %5.1f %%  rank %d/%d" % (r["name"], r["rate"], rows.index(r) + 1, len(rows)))
    print("  plot: %s" % (plot or "unchanged (top 25 same)"))
    print()
    print(git("show", "--stat", "--format=", "HEAD", cwd=args.worktree))
    print(git("diff", "HEAD~1", "--", "README.md", cwd=args.worktree))


if __name__ == "__main__":
    main()
