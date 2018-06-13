#!/bin/bash

for filename in VID_*.final; do
    if [ ! -f "${filename##*/}.processed" ]; then
        echo "running for $filname"
        python processor.py -f "$filename" > "$filename.processed"
    fi
    echo "finished $filename"
done
