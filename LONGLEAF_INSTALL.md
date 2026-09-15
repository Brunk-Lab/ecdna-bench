# Installing ecdna-bench on UNC Longleaf

This page has been replaced by
[`docs/TUTORIAL_LONGLEAF.md`](docs/TUTORIAL_LONGLEAF.md), which covers account
setup, the shared environment, cluster jobs, notebooks and troubleshooting for
Brunk Lab members.

In short:

```bash
bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh
# then, in a new terminal, start every session with
ecdna
```

If you ran the setup before 15 September 2026, run the script once more with
`--migrate`: the earlier version changed settings for all of your conda
environments.

Outside UNC, follow [`docs/TUTORIAL_EXTERNAL.md`](docs/TUTORIAL_EXTERNAL.md).
