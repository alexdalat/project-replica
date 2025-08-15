#!/bin/bash

cat <( sqlite3 "$1" .dump | grep "^ROLLBACK" -v ) <( echo "COMMIT;" ) | sqlite3 "fix_$1"

# now overwrite the original file
mv "fix_$1" "$1"
