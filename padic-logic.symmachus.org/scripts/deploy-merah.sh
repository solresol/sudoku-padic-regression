#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

host="${MERAH_HOST:-merah}"
site="${MERAH_SITE:-padic-logic.symmachus.org}"
user="${MERAH_USER:-padiclogic}"
target="/var/www/vhosts/${site}/htdocs/"

npm run build

rsync -avz --delete \
  --exclude='.DS_Store' \
  dist/ "${user}@${host}:${target}"

echo "Deployed ${site} to ${user}@${host}:${target}"
