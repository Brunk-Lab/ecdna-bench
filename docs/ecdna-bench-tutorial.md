# ecdna-bench tutorial (replaced)

This tutorial has been replaced by two guides:

- [`TUTORIAL_LONGLEAF.md`](TUTORIAL_LONGLEAF.md): Brunk Lab members on UNC Longleaf
  (account setup, the shared environment, cluster jobs, notebooks, troubleshooting).
- [`TUTORIAL_EXTERNAL.md`](TUTORIAL_EXTERNAL.md): everyone else, with or without a GPU.

If you ran the lab setup before 15 September 2026, run it once more with `--migrate`:

```bash
bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh --migrate
```

The earlier version changed settings for all of your conda environments. The new one
only adds the `ecdna` command, and your settings change only in a terminal where you
type it.
