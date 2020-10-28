import numpy as np
import glob

directory = "../../02_real/00_raw/00_timestamps/"

sum = 0
for i in range(83):
    j = "%04d" % (2 * i)
    k = "%04d" % (2 * i + 1)

    files = glob.glob("../../02_real/00_raw/00_timestamps/%s/*.txt" % j)
    start = 0
    for file in files:
        f = open(file, "r")
        time = int(f.readline())

        if time < start or start == 0:
            start = time

    files = glob.glob("../../02_real/00_raw/00_timestamps/%s/*.txt" % k)
    end = 0
    for file in files:
        f = open(file, "r")
        time = int(f.read().splitlines()[-1])

        if time > end:
            end = time

    diff = int(end) - int(start)
    if diff < 0:
        diff = int(start) - int(end)
    sum += diff

print(sum / 83000)