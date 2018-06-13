import glob
import argparse
import os
import datetime
import time

ap = argparse.ArgumentParser()
ap.add_argument("-s", required=True, help="start time")
ap.add_argument("-e", required=True, help="end time")
args = vars(ap.parse_args())

startnum = args["s"]
endnum = args["e"]

def num_to_time(numstr):
    hour = numstr[:2]
    minute = numstr[2:4]
    second = numstr[4:6]
    millis = numstr[6:]

    time = datetime.datetime.now()
    return time.replace(hour=int(hour), minute=int(minute), second=int(second), microsecond=int(millis)*1000)

start = num_to_time(startnum)
end = num_to_time(endnum)

def write(string):
    os.system("xdotool type " + string)
    time.sleep(0.05)

def enter(n = 1):
    for i in range(0, n):
        os.system("xdotool key Return")
        time.sleep(0.05)

def unenter(n = 1):
    for i in range(0, n):
        os.system("xdotool keydown Control")
        os.system("xdotool key Up")
        os.system("xdotool keyup Control")
        time.sleep(0.05)

def tab(n = 1):
    for i in range(0, n):
        os.system("xdotool key Tab")
        time.sleep(0.05)

def untab(n = 1):
    for i in range(0, n):
        os.system("xdotool keydown Shift")
        os.system("xdotool key Tab")
        os.system("xdotool keyup Shift")
        time.sleep(0.05)

files = sorted(glob.glob("VID*.processed"))
for file in files:
    filenum = file.split('_')[-1].split('.')[0]
    filetime = num_to_time(filenum)

    if filetime > start and filetime < end:

        unenter()
        title = file.split('.')[0]
        write(title)

        for line in open(file).read().split('\n'):
            if len(line) > 0:
                mm, fc = line.split(', ')

                enter()
                write(mm)

                tab()
                write(fc)

                untab()
        tab(2)
        time.sleep(1)









