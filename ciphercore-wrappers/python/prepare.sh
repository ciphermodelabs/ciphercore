#!/bin/bash
set -e

if [[ $OSTYPE == 'darwin'* ]]; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  brew install openssl
else
  yum install -y openssl openssl-devel curl
fi
curl https://sh.rustup.rs -sSf > install_cargo.sh
sh install_cargo.sh -y
source $HOME/.cargo/env
cargo build --release
mkdir -p ciphercore-wrappers/python/lib
cp -f target/release/libcadapter.a ciphercore-wrappers/python/lib/libcadapter.a

