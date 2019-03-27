#!/bin/bash

INPUT_FOLDER=$1
INPUT_FOLDER_1080="$1/1080p"
INPUT_FOLDER_720="$1/720p"
INPUT_FOLDER_480="$1/480p"
INPUT_FOLDER_360="$1/360p"
INPUT_FOLDER_240="$1/240p"
INPUT_FOLDER_144="$1/240p"
OUTPUT_SUFFIX=".out"

OUTPUT_FOLDER=$2

export INPUT_FOLDER
export OUTPUT_FOLDER
export INPUT_FOLDER_1080
export INPUT_FOLDER_720
export INPUT_FOLDER_480
export INPUT_FOLDER_360
export INPUT_FOLDER_240
export OUTPUT_SUFFIX

do_process(){
    resolution="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $2 2>&1 | tr x :)"
    ffmpeg -i $1 -i $2 -lavfi "[1:v]split[o1][o2];[0:v]scale=${resolution}[d];[d]split[d1][d2];[o1][d1]ssim=$3;[o2][d2]psnr=$4" -f null - &> $OUTPUT_FOLDER/outssimpsnr.log
}

create_folders(){
    mkdir -p $OUTPUT_FOLDER/ssim/720/$1
    mkdir -p $OUTPUT_FOLDER/ssim/480/$1
    mkdir -p $OUTPUT_FOLDER/ssim/360/$1
    mkdir -p $OUTPUT_FOLDER/ssim/240/$1
    mkdir -p $OUTPUT_FOLDER/ssim/144/$1

    mkdir -p $OUTPUT_FOLDER/psnr/720/$1
    mkdir -p $OUTPUT_FOLDER/psnr/480/$1
    mkdir -p $OUTPUT_FOLDER/psnr/360/$1
    mkdir -p $OUTPUT_FOLDER/psnr/240/$1
    mkdir -p $OUTPUT_FOLDER/psnr/144/$1
}

export -f do_process
export -f create_folders

for main_file in $INPUT_FOLDER_1080/* ; do
    echo "Input file $main_file  `date`"
    filenamewithextensiom=$(basename -- "$main_file")
    filenamewithoutextension="${filenamewithextensiom%.*}"
    bash -c "create_folders \"$filenamewithoutextension\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720/$filenamewithextensiom\" \"$OUTPUT_FOLDER/ssim/720/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\" \"$OUTPUT_FOLDER/psnr/720/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480/$filenamewithextensiom\" \"$OUTPUT_FOLDER/ssim/480/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\" \"$OUTPUT_FOLDER/psnr/480/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360/$filenamewithextensiom\" \"$OUTPUT_FOLDER/ssim/360/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\" \"$OUTPUT_FOLDER/psnr/360/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240/$filenamewithextensiom\" \"$OUTPUT_FOLDER/ssim/240/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\" \"$OUTPUT_FOLDER/psnr/240/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144/$filenamewithextensiom\" \"$OUTPUT_FOLDER/ssim/144/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\" \"$OUTPUT_FOLDER/psnr/144/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    echo "Finished file $main_file `date`"
done
