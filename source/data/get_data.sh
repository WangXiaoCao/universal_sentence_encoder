#!/bin/bash

quora_fid="0B8ZGlkqDw7hFU2I1Vkp4NWduSTA"
glove_fid="0B8ZGlkqDw7hFbmdId3JDY0YwY2M"

# download quora dataset split.
mkdir quora
id=`curl -c cookie.txt -s -L "https://drive.google.com/uc?export=download&id=$quora_fid" | grep confirm | sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/"`
curl -b cookie.txt -L -o quora/quora.zip "https://drive.google.com/uc?confirm=$id&export=download&id=$quora_fid"
rm cookie.txt
unzip quora/quora.zip -d quora
rm quora/quora.zip

# download pretrained embeddings only for quora dataset.
mkdir glove
id=`curl -c cookie.txt -s -L "https://drive.google.com/uc?export=download&id=$glove_fid" | grep confirm | sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/"`
curl -b cookie.txt -L -o glove/glove.zip "https://drive.google.com/uc?confirm=$id&export=download&id=$glove_fid"
rm cookie.txt
unzip glove/glove.zip -d glove
rm glove/glove.zip
