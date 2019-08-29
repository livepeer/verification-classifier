#!/bin/bash

totalFiles=30
originals="gs://livepeer-verifier-originals/"
renditions="gs://livepeer-verifier-renditions/"
datasetPath="../../stream/"
original="1080p/"
count=0
total=0

for filenamePath in $(gsutil ls $originals)
    do

        filename="${filenamePath##*/}"
        found=true
        for rendition in  $(gsutil ls $renditions)
            do
                renditionPath=${rendition#*$renditions}
                if ! [[ $(gsutil -q stat $rendition$filename; echo $?) ]]; then
                    found=false
                    echo "$rendition$filename" not found
                    exit
                fi
        done;

        if [ found  ]; then

            gsutil cp "$originals$filename" "$datasetPath$original"

            ((total++))

            for rendition in  $(gsutil ls $renditions)
                do
                    renditionPath=${rendition#*$renditions}
                    echo "downloading" $rendition
                    gsutil cp "$rendition$filename" "$datasetPath$renditionPath"

            done;

        echo "================== Total files downloaded: $total =================="
        fi

        if [ $total == $totalFiles  ]; then
            exit
        fi
done;