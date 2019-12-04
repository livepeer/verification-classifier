#!/bin/bash

for filename in $(gsutil ls gs://livepeer-qoe-sources/vimeo)
do
    # Define a timestamp function
 
    filename="${filename##*/}"
    gcloud functions call dataset_generator_brisque_http --data '{"name":"'"$filename"'"}' &

    echo "$(date)" "$filename";
    sleep 10
done;