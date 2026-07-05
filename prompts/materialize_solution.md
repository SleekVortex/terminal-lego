Before marking the task complete, write an executable, non-interactive,
idempotent Bash script to /logs/artifacts/solve.sh. Its first line must be
#!/usr/bin/env bash. The script must reproduce all successful changes from a
fresh copy of the task image. It may depend only on files present in that fresh
image and on network access permitted by the task. It must not read /tests,
/solution, /logs/verifier, reward files, or any hidden reference solution. It
must exit non-zero on failure. Finally run:
chmod +x /logs/artifacts/solve.sh

