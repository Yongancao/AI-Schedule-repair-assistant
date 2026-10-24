# Schedule Repair Assistant

Can an LLM build a valid daily schedule, and can it *repair* one when the day
changes? This project compares four ways of doing it, all judged by the same
independent checker:

| Method | Idea |
|---|---|
| LLM only | Ask the model for a schedule once |
| LLM + checker feedback | Show the model the checker's error messages and let it retry |
| Solver | Exact state-space search (no LLM) |
| Hybrid | LLM turns plain-English requirements into a structured problem, the solver solves it |

For repair, a change happens after a plan exists (new meeting, new task,
longer task, shorter day, tighter deadline) and methods are compared on
validity, number of tasks moved, and effort.

## Status

- [x] Problem format, checker, exact solver, minimal-change repair
- [x] Random problem and change generator, with difficulty presets
- [x] Tests, including comparison against brute force
- [ ] LLM wrapper (Gemini) with caching and retries
- [ ] Methods and experiment runner
- [ ] Analysis, graphs, Streamlit demo

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Try it

```powershell
python demo.py              # generate -> solve -> change -> repair
python demo.py 8 hard       # another seed / difficulty
python -m unittest          # run all tests (about 15 seconds)
```

## Project layout

```
scheduler/
  problem.py     data format (Problem, Task, Block), plan helpers
  checker.py     check_plan(): every violation, in plain English
  solver.py      solve() and repair(): exact search
  generator.py   random problems (planted solutions) and changes
tests/           unit tests + brute-force verification
experiments/     experiment scripts (coming next)
results/         CSV outputs and graphs
demo.py          end-to-end tour of the core
```

## Design decisions

- **Time** is an integer number of minutes after midnight. JSON stays exact
  and the checker never has to parse text.
- **One person, one day.** Tasks are single continuous blocks of *exactly*
  their duration. Fixed events ("blocks") cannot be overlapped.
- **Hard constraints only:** availability window, fixed events, deadlines,
  earliest-start times, prerequisites. "Preferences" are represented by the
  repair objective instead (change as little as possible).
- **The checker is the ground truth.** It is written independently of the
  solver, and reports a specific message per violation. Those messages are
  also the feedback given to the LLM in the feedback-loop method.
- **Solvable by construction.** Generated problems start from a hidden valid
  schedule, so failures are always the method's fault. Changes can be
  required to keep the problem solvable and to break the old plan.
- **Repair objective:** minimize the *number of tasks whose start time
  changes*. This is exact. Total minutes shifted is reported too but is not
  optimized, so a repair can move few tasks by a large amount; comparing the
  two metrics is part of the analysis.
- **Solver:** depth-first search over task sequences with left-shift
  placement, dominance pruning, and branch and bound. Exact and fast for the
  problem sizes used here (up to about 15 tasks).

## Verification

`tests/test_solver.py` compares the solver against exhaustive search on
small problems: feasibility must match exactly, and repair must move exactly
the minimum possible number of tasks. The comparison covers feasible and
infeasible changes of every kind.

## AI assistance

This project was built for a university AI course with help from Claude
(Anthropic), as permitted by the course policy. The code was reviewed and
tested by the author. *(Edit this note to describe your own contribution.)*
