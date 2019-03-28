import pandas as pd 
import numpy as np
import zipfile

def trimList(originalList):
    newTripList = []
    for item in originalList[:]:
        newTripList.append(item.strip())    
    return newTripList

def mapColumnToNum(dataFrame,columnName):
    dfColsArray = pd.unique(dataFrame[columnName])
    dfColsList = dfColsArray.tolist()
    newTripList = trimList(dfColsList)
    dfCols = dataFrame[columnName]
    dfCols = dfCols.map(lambda eachStr: newTripList.index(eachStr.strip()))
    return dfCols

#unzip a zip file to current folder *.zip
def unzipFile(zipFile):
    zfile = zipfile.ZipFile(zipFile,'r')
    for filename in zfile.namelist():
      data = zfile.read(filename)
      file = open(filename, 'w+b')
      file.write(data)
      file.close()    
    return 0

import os
import glob

#Delete files with path/name pattern, eg:  deleteFileByPattern("./RentListingInquries*")
def deleteFileByPattern(fileNamePattern):
    for fileName in glob.glob(fileNamePattern):
       print(fileName)
       os.remove(fileName)
    return 1

#append one column with default value. 
def appendColumn(df,columnName, defaultValue=0):
    df[columnName] = defaultValue
    return df

