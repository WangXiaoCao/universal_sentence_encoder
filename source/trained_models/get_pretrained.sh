#!/bin/bash

pretrained='https://drive.google.com/uc?export=download&id=0B8ZGlkqDw7hFSm1MQ2FDVTZCTjA'

# download pretrained models.
curl -o pretrained_models.zip $pretrained
unzip pretrained_models.zip
rm pretrained_models.zip
