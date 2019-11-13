#!/bin/bash

for filename in $(gsutil ls gs://livepeer-verifier-originals/)
do
    # Define a timestamp function
 
    filename="${filename##*/}"
    gcloud functions call dataset_generator_http --data '{"name":"'"$filename"'", "resolution_list": "1080p,144p"}' &

    echo "$(date)" "$filename" "1080p,144p";
    sleep 10

    gcloud functions call dataset_generator_http --data '{"name":"'"$filename"'", "resolution_list": "720p,240p"}' &
    echo "$(date)" "$filename" "720,240p";
    sleep 10

    gcloud functions call dataset_generator_http --data '{"name":"'"$filename"'", "resolution_list": "480p,360p"}' &
    echo "$(date)" "$filename" "480p,360p";
    sleep 10
done;