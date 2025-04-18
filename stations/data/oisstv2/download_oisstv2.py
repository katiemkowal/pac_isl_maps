#script to download oisstv2 sea surface temperature data
import requests
#from pathlib import Path 
import os
from os import path
from datetime import datetime

current_year = datetime.now().year

first_year = 1981
last_year = 2025

working_dir = '/cpc/int_desk/data/oisstv2'

#get a list of year strings from a start and an end date
def getYearStrings(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [str(year) for year in range(start, end+1)]

#helper function to download data given a url
def download_url(url):
  # assumes that the last segment after the / represents the file name
  # if the url is http://abc.com/xyz/file.txt, the file name will be file.txt
    print("downloading: ",url)
    file_name_start_pos = url.rfind("/") + 1
    file_name = url[file_name_start_pos:]

    r = requests.get(url, stream=True)
    if r.status_code == requests.codes.ok:
        with open(file_name, 'wb') as f:
            for data in r:
                f.write(data)

#download raw SST observed data given URL for OISST website hosted by NOAA
def download_OISST(DATAFOLDER, WORKINGFOLDER, URL, FIRSTYEAR, LASTYEAR):
    os.chdir(DATAFOLDER)
    YEARS = getYearStrings(FIRSTYEAR, LASTYEAR)
    for y in YEARS:
        target_file = "sst.day.mean." + y + ".nc"
        print(target_file)
        if y == str(current_year):
            if path.exists(target_file):
                os.remove(target_file)
        if not path.exists(target_file):
            download_url(URL + target_file)
        else:
            print("already downloaded")
    os.chdir(WORKINGFOLDER)

#download oisst data from https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/ 
oisst_url = 'https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/'
download_OISST(working_dir, 
               working_dir, 
               oisst_url, 
               first_year,
               last_year)