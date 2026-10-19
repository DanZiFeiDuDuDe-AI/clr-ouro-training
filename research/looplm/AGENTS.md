# CLR experiment constraints and artifact transfer

- Continue only the authorized O3-REBASE2 VGen2 failure-localization work. Do not train an allocator, reconnect router/memory, or run integrated O3 tests. Do not tune on TestA/TestB.
- Keep Candidate B unchanged and Rule A threshold exactly `0.8504285216331482`.
- Check the current artifacts: the archived 10K run passes standalone validation; the pasted 3K failure description is an earlier state. A gate pass alone does not authorize downstream work.
- After **every** completed remote training run (including diagnostic probes), reconnect to the server and pull its checkpoints to the local experiment directory. Verify SHA-256; retain configs, metrics, and a transfer manifest. Do not claim completion while weights are only remote.
- Use `tools/remote_weights.py` for SSH and verified downloads. Credentials come from `LOOPLM_SSH_PASSWORD` or `.local/scnet_password.dpapi`, protected for the current Windows user. Never print credentials or put them in experiment artifacts.
- Do not overwrite a differing local checkpoint. Download a changed version into a separate run directory.
