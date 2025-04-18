############## DIRECTORIES

download_working_dir = '/cpc/int_desk/data/c3s/code' # current working directory
download_raw_data_dir = '/cpc/int_desk/data/c3s/data/raw/' #download directory for raw grib files
download_processed_data_dir = '/cpc/int_desk/data/c3s/data/processed/' #directory for processed files converted to netcdf

############## VARIABLES
vars = ['sst', 'prcp'] #variables of interest

## hindcast years - make sure these align with the available data from the gcms you are downloading - e.g. if you download MeteoFrance, start in 1991
## check availability at 
first_year = 1993 #first year of hindcast data
last_year = 2024 #last year of hindcast data

#months to initialize the models
initialized_months = ['Jan', 'Jul']

#spatial extents - can keep adding onto these, just make sure the variable matches the variable name up in 'vars' - e.g. gph with gph
#the code is setup to read in pressure level data as well, but make sure to add a pressure level variable by name - e.g. 'gph_pressure_level' = 850
prcp_download_extent = {'west': -180, 'east': 180, 'north': 50, 'south': -50} #extent to download the prcp data
prcp_download_name = 'global-tropics' # how you want to name the prcp spatial extent when you save the file

sst_download_extent = {'west': -180, 'east': 180, 'north': 50, 'south': -50} #spatial extent to download the sst data
sst_download_name = 'global-tropics' # how you want to name the sst spatial extent when you save the file

# GCMs to download, the naming conventions are defined in the retrieve_functions.py script, can add on more models as needed
gcms = ['ECMWF-SEAS5.1', 'ECCC-CANSIPS-IC3', 'CMCC-SPSV3.5', 'METEOFRANCE-SYSTEM8', 'DWD-GCFSV2.1']
#options defined in retrieve_functions dictionary - some included are['ECMWF-SEAS5.1','ECCC-CANSIPS-IC3','DWD-GCFSV2.1','CMCC-SPSV3.5', 'METEOFRANCE-SYSTEM8','NCEP-CFSV2','UKMO-GLOSEA601']
#ncep cfsv2 missing 2017, 2018, 2019
#cmcc 35 mssing 2017, 2018, 2019, 2020
#meteo france system 8 missing 2019, 2020
#glosea603 is has up to 2016 and 2024
#glosea602 has up to 2016 and 2023 and 2024
#glosea601 has up to 2016 and 2022 and 2023
#glosea600 has up to 2016 and 2021 and 2022
#dwd 21 is missing 2020

#lead times - what lead times you want counting out from initialization date - ['1', '2', '3', '4', '5', '6'] options for cds and iri - but iri can extent longer if desired
download_type = 'daily_single'
leadtimes = ["168","336","504","672"]# one month leadtime refers to the month you are in, so 31 days for cds and 0.5 lead time for iri, a 1.5 month lead in iri dict is 2months here