---
name: Pull Request
about: Submit a change to the project
---

## Summary

One or two sentences on what this PR does and why.

## Related issue

Fixes #___ / Relates to #___ / None.

## Type of change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New alpha template
- [ ] New gate rule or metric
- [ ] Performance improvement
- [ ] Documentation / CI / tooling change

## Checklist

- [ ] `python -m pytest tests/ -v` passes locally
- [ ] `python -m mypy brainlib/ scripts/ tests/` passes locally
- [ ] I added a regression test for any behavior I changed
- [ ] I didn't duplicate logic — I extended `brainlib/` instead of
      copy-pasting into a script
- [ ] If I touched the harness, the mock branch still works
- [ ] I updated `CHANGELOG.md` under `[Unreleased]`

## Test output

Paste the relevant pytest / mypy summary lines here.
