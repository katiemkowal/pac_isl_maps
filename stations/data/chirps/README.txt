This folder includes data downloaded from UCSB website at .25' and 0.05' resolution
The download_chirps scripts can be run by activating a python environment and running the command
#for higher res
$python download_chirps-5km.py
#for lower res
$python download_chirps-25km.py

This script will automatically search for missing years between 1981- the current year (calculates the current year in the script), and then delete the current year and redownload to get more recent days if available on the web page

The script can be modified to change the years for downloading

The script can also be modified to download data from other links, e.g. lower resolution chirps data links, or others, just make sure you specify the file name, new url, and place this file in the folder in which you want to download the new data (current setup is to download from a url into the current file folder).