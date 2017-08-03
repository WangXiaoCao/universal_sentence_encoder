#!/bin/bash

quora='https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFU2I1Vkp4NWduSTA'
glovepath='https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFbmdId3JDY0YwY2M'

# download quora dataset split.
mkdir quora
curl -Lo quora/quora.zip $quora
unzip quora/quora.zip -d quora
rm quora/quora.zip


# download pretrained embeddings only for quora dataset.
mkdir glove
curl -Lo glove/glove.zip $glovepath
unzip glove/glove.zip -d glove
rm glove/glove.zip
