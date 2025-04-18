import os
from os import path
import requests
from datetime import datetime

#get a list of year strings from a start and an end date
def getYearStrings(start, end):
    #start year, to end year inclusive
    # >get YearStrings ("1981", "1982"... "2016")
    #return [expression for var is iterable if condition]
    return [str(year) for year in range(start, end+1)]

#downloads data from a given url
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

def download_CHIRPS(URL, FIRSTYEAR, LASTYEAR):
    YEARS = getYearStrings(FIRSTYEAR, LASTYEAR)
    for y in YEARS:
        ### update this if other file names are desired
        target_file = "chirps-v2.0." + y + ".days_p05.nc"
        print(target_file)
        if not path.exists(target_file):
            download_url(URL + target_file)
        else:
            print("already downloaded")

#url to download the data
chirps_url = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05/'

current_year = datetime.now().year
print('current year is ' + str(current_year))
os.remove("chirps-v2.0." + str(current_year) + ".days_p05.nc")
#to get most updated data, remove current year and redownload up to current year

#note this function will download the data into whatever directory this python file is located
download_CHIRPS(chirps_url, 1981, current_year)