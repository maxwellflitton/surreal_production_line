#!/usr/bin/env bash

cd /Users/maxwellflitton/Documents/github/surrealDB/patch_22_11_2023/surrealdb

cargo run -- start --log trace --user root --pass root memory
