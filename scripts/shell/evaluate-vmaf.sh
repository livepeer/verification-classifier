#!/bin/bash

INPUT_FOLDER=$1
INPUT_FOLDER_1080="$1/1080p"
INPUT_FOLDER_720="$1/720p"
INPUT_FOLDER_480="$1/480p"
INPUT_FOLDER_360="$1/360p"
INPUT_FOLDER_240="$1/240p"
INPUT_FOLDER_144="$1/144p"
OUTPUT_SUFFIX=".out"

INPUT_FOLDER_1080_WATERMARK="$1/1080p_watermark"
INPUT_FOLDER_720_WATERMARK="$1/720p_watermark"
INPUT_FOLDER_480_WATERMARK="$1/480p_watermark"
INPUT_FOLDER_360_WATERMARK="$1/360p_watermark"
INPUT_FOLDER_240_WATERMARK="$1/240p_watermark"
INPUT_FOLDER_144_WATERMARK="$1/144p_watermark"

INPUT_FOLDER_1080_FLIP_VERTICAL="$1/1080p_flip_vertical"
INPUT_FOLDER_720_FLIP_VERTICAL="$1/720p_flip_vertical"
INPUT_FOLDER_480_FLIP_VERTICAL="$1/480p_flip_vertical"
INPUT_FOLDER_360_FLIP_VERTICAL="$1/360p_flip_vertical"
INPUT_FOLDER_240_FLIP_VERTICAL="$1/240p_flip_vertical"
INPUT_FOLDER_144_FLIP_VERTICAL="$1/144p_flip_vertical"

INPUT_FOLDER_1080_FLIP_HORIZONTAL="$1/1080p_flip_horizontal"
INPUT_FOLDER_720_FLIP_HORIZONTAL="$1/720p_flip_horizontal"
INPUT_FOLDER_480_FLIP_HORIZONTAL="$1/480p_flip_horizontal"
INPUT_FOLDER_360_FLIP_HORIZONTAL="$1/360p_flip_horizontal"
INPUT_FOLDER_240_FLIP_HORIZONTAL="$1/240p_flip_horizontal"
INPUT_FOLDER_144_FLIP_HORIZONTAL="$1/144p_flip_horizontal"

INPUT_FOLDER_1080_ROTATE_90_CLOCKWISE="$1/1080p_rotate_90_clockwise"
INPUT_FOLDER_720_ROTATE_90_CLOCKWISE="$1/720p_rotate_90_clockwise"
INPUT_FOLDER_480_ROTATE_90_CLOCKWISE="$1/480p_rotate_90_clockwise"
INPUT_FOLDER_360_ROTATE_90_CLOCKWISE="$1/360p_rotate_90_clockwise"
INPUT_FOLDER_240_ROTATE_90_CLOCKWISE="$1/240p_rotate_90_clockwise"
INPUT_FOLDER_144_ROTATE_90_CLOCKWISE="$1/144p_rotate_90_clockwise"

INPUT_FOLDER_1080_ROTATE_90_COUNTERCLOCKWISE="$1/1080p_rotate_90_counterclockwise"
INPUT_FOLDER_720_ROTATE_90_COUNTERCLOCKWISE="$1/720p_rotate_90_counterclockwise"
INPUT_FOLDER_480_ROTATE_90_COUNTERCLOCKWISE="$1/480p_rotate_90_counterclockwise"
INPUT_FOLDER_360_ROTATE_90_COUNTERCLOCKWISE="$1/360p_rotate_90_counterclockwise"
INPUT_FOLDER_240_ROTATE_90_COUNTERCLOCKWISE="$1/240p_rotate_90_counterclockwise"
INPUT_FOLDER_144_ROTATE_90_COUNTERCLOCKWISE="$1/144p_rotate_90_counterclockwise"

OUTPUT_FOLDER=$2

export INPUT_FOLDER
export OUTPUT_FOLDER
export INPUT_FOLDER_1080
export INPUT_FOLDER_720
export INPUT_FOLDER_480
export INPUT_FOLDER_360
export INPUT_FOLDER_240
export OUTPUT_SUFFIX

export INPUT_FOLDER_1080_WATERMARK
export INPUT_FOLDER_720_WATERMARK
export INPUT_FOLDER_480_WATERMARK
export INPUT_FOLDER_360_WATERMARK
export INPUT_FOLDER_240_WATERMARK
export INPUT_FOLDER_144_WATERMARK

export INPUT_FOLDER_1080_FLIP_VERTICAL
export INPUT_FOLDER_720_FLIP_VERTICAL
export INPUT_FOLDER_480_FLIP_VERTICAL
export INPUT_FOLDER_360_FLIP_VERTICAL
export INPUT_FOLDER_240_FLIP_VERTICAL
export INPUT_FOLDER_144_FLIP_VERTICAL

export INPUT_FOLDER_1080_FLIP_HORIZONTAL
export INPUT_FOLDER_720_FLIP_HORIZONTAL
export INPUT_FOLDER_480_FLIP_HORIZONTAL
export INPUT_FOLDER_360_FLIP_HORIZONTAL
export INPUT_FOLDER_240_FLIP_HORIZONTAL
export INPUT_FOLDER_144_FLIP_HORIZONTAL

export INPUT_FOLDER_1080_ROTATE_90_CLOCKWISE
export INPUT_FOLDER_720_ROTATE_90_CLOCKWISE
export INPUT_FOLDER_480_ROTATE_90_CLOCKWISE
export INPUT_FOLDER_360_ROTATE_90_CLOCKWISE
export INPUT_FOLDER_240_ROTATE_90_CLOCKWISE
export INPUT_FOLDER_144_ROTATE_90_CLOCKWISE

export INPUT_FOLDER_1080_ROTATE_90_COUNTERCLOCKWISE
export INPUT_FOLDER_720_ROTATE_90_COUNTERCLOCKWISE
export INPUT_FOLDER_480_ROTATE_90_COUNTERCLOCKWISE
export INPUT_FOLDER_360_ROTATE_90_COUNTERCLOCKWISE
export INPUT_FOLDER_240_ROTATE_90_COUNTERCLOCKWISE
export INPUT_FOLDER_144_ROTATE_90_COUNTERCLOCKWISE

do_process(){
    resolution="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 $2 2>&1 | tr x :)"
    ffmpeg -i $1 -i $2 -lavfi "scale=${resolution}, libvmaf" -f null - 1> $3 2> $3$OUTPUT_SUFFIX
}

create_folders(){
    mkdir -p $OUTPUT_FOLDER/vmaf/720/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144/$1

    mkdir -p $OUTPUT_FOLDER/vmaf/1080_watermark/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/720_watermark/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480_watermark/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360_watermark/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240_watermark/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144_watermark/$1

    mkdir -p $OUTPUT_FOLDER/vmaf/1080_flip_vertical/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/720_flip_vertical/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480_flip_vertical/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360_flip_vertical/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240_flip_vertical/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144_flip_vertical/$1

    mkdir -p $OUTPUT_FOLDER/vmaf/1080_flip_horizontal/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/720_flip_horizontal/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480_flip_horizontal/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360_flip_horizontal/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240_flip_horizontal/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144_flip_horizontal/$1

    mkdir -p $OUTPUT_FOLDER/vmaf/1080_rotate_90_clockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/720_rotate_90_clockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480_rotate_90_clockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360_rotate_90_clockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240_rotate_90_clockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144_rotate_90_clockwise/$1

    mkdir -p $OUTPUT_FOLDER/vmaf/1080_rotate_90_counterclockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/720_rotate_90_counterclockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/480_rotate_90_counterclockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/360_rotate_90_counterclockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/240_rotate_90_counterclockwise/$1
    mkdir -p $OUTPUT_FOLDER/vmaf/144_rotate_90_counterclockwise/$1

}

export -f do_process
export -f create_folders

for main_file in $INPUT_FOLDER_1080/* ; do
    echo "Input file $main_file  `date`"
    filenamewithextensiom=$(basename -- "$main_file")
    filenamewithoutextension="${filenamewithextensiom%.*}"
    bash -c "create_folders \"$filenamewithoutextension\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_1080_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/1080_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_1080.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144_WATERMARK/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144_watermark/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_1080_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/1080_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_1080.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144_FLIP_VERTICAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144_flip_vertical/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_1080_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/1080_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_1080.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144_FLIP_HORIZONTAL/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144_flip_horizontal/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_1080_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/1080_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_1080.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144_ROTATE_90_CLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144_rotate_90_clockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_1080_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/1080_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_1080.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_720_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/720_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_720.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_480_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/480_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_480.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_360_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/360_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_360.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_240_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/240_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_240.log\""
    bash -c "do_process \"$main_file\" \"$INPUT_FOLDER_144_ROTATE_90_COUNTERCLOCKWISE/$filenamewithextensiom\" \"$OUTPUT_FOLDER/vmaf/144_rotate_90_counterclockwise/$filenamewithoutextension/$filenamewithoutextension\"\"_144.log\""

    echo "Finished file $main_file `date`"
done
