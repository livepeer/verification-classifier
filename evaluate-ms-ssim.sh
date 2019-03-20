#!/bin/bash

INPUT_FOLDER=$1
INPUT_FOLDER_1080="$1/1080p"
INPUT_FOLDER_720="$1/720p"
INPUT_FOLDER_480="$1/480p"
INPUT_FOLDER_360="$1/360p"
INPUT_FOLDER_240="$1/240p"
INPUT_FOLDER_144="$1/240p"

export INPUT_FOLDER
export OUTPUT_FOLDER
export INPUT_FOLDER_1080
export INPUT_FOLDER_720
export INPUT_FOLDER_480
export INPUT_FOLDER_360
export INPUT_FOLDER_240
export INPUT_FOLDER_144

do_process(){
    resolution="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $2 2>&1 | tr x :)"
    avconv -i $1 -i $2 -filter_complex "scale=${resolution}, measure=$3" -f null - &> out.log
}

create_folders(){
    mkdir -p output/mssim/720/$1
    mkdir -p output/mssim/480/$1
    mkdir -p output/mssim/360/$1
    mkdir -p output/mssim/240/$1
    mkdir -p output/mssim/144/$1
}

export -f do_process
export -f create_folders

for main_file in $INPUT_FOLDER_1080/* ; do
    echo "Input file $main_file  `date`"
    filenamewithextensiom=$(basename -- "$main_file")
    filenamewithoutextension="${filenamewithextensiom%.*}"
    bash -c "create_folders \"$filenamewithoutextension\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720/$filenamewithextensiom\" \"output/mssim/720/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480/$filenamewithextensiom\" \"output/mssim/480/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360/$filenamewithextensiom\" \"output/mssim/360/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240/$filenamewithextensiom\" \"output/mssim/240/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144/$filenamewithextensiom\" \"output/mssim/144/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    echo "Finished file $main_file `date`"
done
