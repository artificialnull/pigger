#!/bin/bash
for filename in Pig\ study\ 2018-05-31/VID_*.mp4; do
    if [ ! -f "output/${filename##*/}.final.processed" ]; then
        mpv "$filename"
    fi
    echo "finished $filename"
    sleep 0.4
done
