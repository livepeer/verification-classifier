#!/bin/bash

mkdir scripts
cp -r ../scripts/* scripts
docker build -t epicjupiter:v1 .
rm -rf scripts