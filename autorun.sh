#!/bin/bash

for filename in Pig\ study\ 2018-05-31/VID_*.mp4; do
    if [ ! -f "output/${filename##*/}.final" ]; then
        python ruler.py -f "$filename"
        cd output/
        python processor.py -f "${filename##*/}.final" > "${filename##*/}.final.processed"
        cd ..
    fi
    echo "finished $filename"
    sleep 0.4
done
