#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/app"
LOG_DIR="${APP_DIR}/logs"

mkdir -p "${LOG_DIR}"

# Timezone
TZ_VALUE="${TZ:-America/New_York}"
if [ -f "/usr/share/zoneinfo/${TZ_VALUE}" ]; then
  ln -snf "/usr/share/zoneinfo/${TZ_VALUE}" /etc/localtime
  echo "${TZ_VALUE}" > /etc/timezone
fi

# Cron schedule
# Default: run after market close (16:15 ET, Mon-Fri)
SCHEDULE="${FLAGSHIP_CRON_SCHEDULE:-15 16 * * 1-5}"
CMD="${FLAGSHIP_CRON_COMMAND:-/app/flagship/paper_trading/run_full_daily_cycle.sh}"

# Optional secondary cron for portfolio heartbeat (e.g., every 5 min during RTH)
SCHEDULE_PORTFOLIO="${FLAGSHIP_CRON_SCHEDULE_PORTFOLIO:-}"
CMD_PORTFOLIO="${FLAGSHIP_CRON_COMMAND_PORTFOLIO:-}"

CRON_FILE="/etc/cron.d/flagship"
{
  echo "SHELL=/bin/bash"
  echo "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  echo "TZ=${TZ_VALUE}"
  echo "${SCHEDULE} root ${CMD} >> ${LOG_DIR}/cron.log 2>&1"
  if [ -n "${SCHEDULE_PORTFOLIO}" ] && [ -n "${CMD_PORTFOLIO}" ]; then
    echo "${SCHEDULE_PORTFOLIO} root ${CMD_PORTFOLIO} >> ${LOG_DIR}/cron.log 2>&1"
  fi
} > "${CRON_FILE}"

chmod 0644 "${CRON_FILE}"
crontab "${CRON_FILE}"

echo "[entrypoint] TZ=${TZ_VALUE}"
echo "[entrypoint] CRON=${SCHEDULE} ${CMD}"
echo "[entrypoint] Starting cron (foreground)..."

exec cron -f


