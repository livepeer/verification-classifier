#!/bin/bash
# Creates all renditions from source videos. Maintain folder structure like below.
input_raw="../../../data/renditions/1080p"
output="../../../data/renditions/"
metadata="../../YT8M_downloader/yt8m_data.csv"
rescaled_input="../../../data/renditions/"
python encode_renditions.py -i $input_raw -o $output -m $metadata
python chroma_subsampling.py -i "$input_raw" -o "$output" -m "$metadata" -s yuv422p
python black_and_white.py -i $rescaled_input  -o $output
python flip.py -i $rescaled_input -o $output -vf
python flip.py -i $rescaled_input -o $output -hf
python flip.py -i $rescaled_input -o $output -cf
python flip.py -i $rescaled_input -o $output -ccf
python low_bitrate.py -i "$input_raw" -o "$output" -m "$metadata" -d 4
python low_bitrate.py -i "$input_raw" -o "$output" -m "$metadata" -d 8
python watermark.py -i "$input_raw" -o "$output" -m "$metadata" -w watermark/livepeer-690x227.png
python watermark.py -i "$input_raw" -o "$output" -m "$metadata" -s 345x114 -w watermark/livepeer-345x114.png
python watermark.py -i "$input_raw" -o "$output" -m "$metadata" -s 856x856 -w watermark/lpt-856x856.png
python vignette.py -i $rescaled_input -o $output