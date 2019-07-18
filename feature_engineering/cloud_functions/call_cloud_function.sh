#!/bin/bash

for filename in $(gsutil ls gs://livepeer-verifier-originals/)
do
    filename="${filename##*/}"
    gcloud beta functions call measure_asset_http --data '{"name":"'"$filename"'"}' &
    echo "$filename launched"
    sleep 7
done;