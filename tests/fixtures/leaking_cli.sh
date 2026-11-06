#!/bin/sh
# Deliberate positive control: these assertions must reject this output.
printf '%s\n' 'deliberate CLI leaking control'
printf 'HF_TOKEN=%s\n' "${HF_TOKEN:?generated marker required}" >&2
printf '%s\n' 'Caused by: deliberate diagnostic chain' >&2
printf '%s\n' 'Stack backtrace:' >&2
exit 23
