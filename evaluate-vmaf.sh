#!/bin/bash

INPUT_FOLDER=$1
INPUT_FOLDER_1080="$1/7"
INPUT_FOLDER_720="$1/6"
INPUT_FOLDER_480="$1/5"
INPUT_FOLDER_360="$1/4"
INPUT_FOLDER_240="$1/3"
OUTPUT_SUFFIX=".out"

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
    ffmpeg -i $1 -i $2 -lavfi "scale=${resolution}, libvmaf" -f null - 1> $3 2> $3$OUTPUT_SUFFIX
}

create_folders(){
    mkdir -p output/vmaf/720/$1
    mkdir -p output/vmaf/480/$1
    mkdir -p output/vmaf/360/$1
    mkdir -p output/vmaf/240/$1
}

export -f do_process
export -f create_folders

for main_file in $INPUT_FOLDER_1080/* ; do
    echo "Input file $main_file  `date`"
    filenamewithextensiom=$(basename -- "$main_file")
    filenamewithoutextension="${filenamewithextensiom%.*}"
    bash -c "create_folders \"$filenamewithoutextension\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720/$filenamewithextensiom\" \"output/vmaf/720/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480/$filenamewithextensiom\" \"output/vmaf/480/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360/$filenamewithextensiom\" \"output/vmaf/360/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240/$filenamewithextensiom\" \"output/vmaf/240/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""

    echo "Finished file $main_file `date`"
done
