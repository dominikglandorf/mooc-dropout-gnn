#!/bin/bash

url="https://snap.stanford.edu/data/act-mooc.tar.gz"

dest_folder="data"

mkdir -p "$dest_folder"

cd "$dest_folder"

wget "$url"

tar -xzf "act-mooc.tar.gz"

rm "act-mooc.tar.gz"

find . -name '._*' -delete

cd -

echo "Succesfully downloaded mooc dataset."
