#! /usr/bin/env python3
# coding: utf-8
"""
This module is put label for FX-history-data.
This label means 0:don't move 1:buy 2:buy full force 3:sell 4:sell full force
"""
import csv
import datetime as dt
import numpy as np

csvfile_r = 'USDJPY_Candlestick_1_h_BID_01.11.2004-31.12.2016.csv'
csvfile_w = 'USDJPY_Candlestick_1_h_Train.csv'

FR = open(csvfile_r, "r", encoding="utf-8")
FW = open(csvfile_w, "w")
reader = csv.reader(FR)
writer = csv.writer(FW, lineterminator='\n')

header = next(reader)
header.append("MvAve")
header.append("Label")
writer.writerow(header)

lineData = []
dayList = []
longList = np.zeros(0)
longAvgList = np.zeros(0)
shortList = np.zeros(0)
for line in reader:
    if line[5] != '0':
        nowDate = dt.datetime.strptime(line[0][0:19], '%d.%m.%Y %H:%M:%S')
        currentAvg = round((float(line[1]) + float(line[4])) / 2, 3)
        longList = np.append(longList, currentAvg)
        dayList.append(nowDate)
        oldestDate = dayList[0]
        if nowDate - dt.timedelta(days=10) >= oldestDate:
            longAvg = round(np.average(longList), 3)
            longAvgList = np.append(longAvgList, longAvg)
            longList = np.delete(longList, 0)
            dayList.pop(0)
            shortList = np.append(shortList, currentAvg)
            lineData.append(line)
        if shortList.size > 10:
            baseAvg = shortList[0]
            shortList = np.delete(shortList, 0)
            shortList = shortList.reshape(2, 5)
            shortAvg5 = np.average(shortList, 1)[0]
            shortAvg10 = np.average(shortList)
            shortList = shortList.reshape(10)
            lineData[0].append(longAvgList[0])
            longAvgList = np.delete(longAvgList, 0)
            if shortAvg5 -baseAvg >= 0.3:
                if shortAvg10 - baseAvg >= 0.3:
                    lineData[0].append(2)
                else:
                    lineData[0].append(1)
            elif shortAvg5 - baseAvg <= -0.3:
                if shortAvg10 - baseAvg <= -0.3:
                    lineData[0].append(4)
                else:
                    lineData[0].append(3)
            else:
                lineData[0].append(0)
            lineData[0][5] = round(float(lineData[0][5]), 3)
            writer.writerow(lineData[0])
            lineData.pop(0)
FR.close()
FW.close()
