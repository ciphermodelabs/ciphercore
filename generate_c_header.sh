#!/bin/sh
cbindgen --config cbindgen.toml --crate ciphercore-adapters --output ciphercore-wrappers/C-wrapper/raw.h
