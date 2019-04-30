#!/bin/bash

mkdir scripts
cp -r ../scripts/* scripts
docker build -t epicjupiter-analytics:v1 .
rm -rf scripts