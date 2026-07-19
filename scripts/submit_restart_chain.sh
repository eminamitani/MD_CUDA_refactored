#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  echo "usage: $0 SEGMENT_COUNT PBS_SCRIPT [INITIAL_DEPENDENCY_JOB_ID]" >&2
  exit 2
fi

segment_count="$1"
pbs_script="$2"
dependency="${3:-}"

if ! [[ "$segment_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "SEGMENT_COUNT must be a positive integer" >&2
  exit 2
fi
test -s "$pbs_script"
command -v jsub >/dev/null

for ((segment = 0; segment < segment_count; ++segment)); do
  if [[ -n "$dependency" ]]; then
    output=$(jsub -W "depend=afterok:${dependency}" "$pbs_script")
  else
    output=$(jsub "$pbs_script")
  fi
  job_id=$(awk 'NF {value=$NF} END {print value}' <<<"$output")
  if [[ -z "$job_id" ]]; then
    echo "could not parse jsub output: $output" >&2
    exit 3
  fi
  printf '%s\t%s\t%s\n' "$segment" "$job_id" "${dependency:-none}"
  dependency="$job_id"
done
