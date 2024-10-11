#!/usr/bin/env bash

curl http://opihi.cs.uvic.ca/sound/genres.tar.gz > genres.tar.gz
tar xvzf genres.tar.gz
mv genres data
rm genres.tar.gz

mkdir -p -v train/{blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}
mkdir -p -v test/{blues,classical,country,disco,hiphop,jazz,metal,pop,reggae,rock}
