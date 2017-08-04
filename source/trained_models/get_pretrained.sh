#!/bin/bash

fid="0B8ZGlkqDw7hFSm1MQ2FDVTZCTjA"

# download pretrained models.
id=`curl -c cookie.txt -s -L "https://drive.google.com/uc?export=download&id=$fid" | grep confirm | sed -e "s/^.*confirm=\(.*\)&amp;id=.*$/\1/"`
curl -b cookie.txt -L -o pretrained_models.zip "https://drive.google.com/uc?confirm=$id&export=download&id=$fid"
rm cookie.txt
unzip pretrained_models.zip
rm pretrained_models.zip
