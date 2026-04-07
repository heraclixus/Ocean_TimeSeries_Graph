#!/bin/bash
for file in *.data; do
  if [ -f "$file" ]; then
    sed -i 's/[[:space:]]\+/ /g' "$file"
  fi
done
