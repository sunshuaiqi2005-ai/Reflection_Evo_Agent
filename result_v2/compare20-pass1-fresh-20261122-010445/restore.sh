#!/bin/sh
set -eu
cd "$(dirname "$0")"
cat archives/compare20-pass1-fresh-20261122-010445-complete.tar.xz.part* > compare20-pass1-fresh-20261122-010445-complete.tar.xz
tar -xJf compare20-pass1-fresh-20261122-010445-complete.tar.xz -C ..
