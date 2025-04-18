############## DIRECTORIES
download_working_dir = '/cpc/int_desk/era5/code'
download_raw_data_dir = '/cpc/int_desk/data/era5/data/raw'

############## VARIABLES
var = 't2m'
first_clim_year = 1991
last_clim_year = 2020
days_back = 14 #how many days back to go

hours_to_download = [ "00:00", #"01:00", "02:00",
        #"03:00", "04:00", "05:00",
        "06:00", #"07:00", "08:00",
        #"09:00", "10:00", "11:00",
        "12:00", #"13:00", "14:00",
        #"15:00", "16:00", "17:00",
        "18:00"#, #"19:00", "20:00",
        #"21:00", "22:00", "23:00"]
                    ]

global_download_extent = {'west': -180, 'east': 180, 'north': 90, 'south': -90} #global
africa_download_extent = {'west': -20, 'east': 55, 'north': 40, 'south':-40} #africa
yemen_download_extent = {'west': 35, 'east': 60, 'north': 25, 'south':12} #yemen
eur_download_extent = {'west': -15, 'east': 45, 'north': 75, 'south':25} #europe
car_download_extent = {'west': -120, 'east': -40, 'north': 35, 'south':0} #caribbean?
cam_download_extent = {'west': -95, 'east': -65, 'north': 25, 'south':2} #central america
nam_download_extent = {'west': -130, 'east': -65, 'north': 60, 'south':15} #north america
sam_download_extent = {'west': -95, 'east': -20, 'north': 15, 'south':-60} #south america
nsam_download_extent = {'west': -90, 'east': -50, 'north': 15, 'south':-20} #northern south america
camnsam_download_extent = {'west': -95, 'east': -65, 'north': 25, 'south':-20} #combined central america northern south america
cas_download_extent = {'west': 40, 'east': 90, 'north': 60, 'south':20} #central asia
sea_download_extent = {'west': 70, 'east': 145, 'north': 60, 'south':5} #southeast asia
mrc_download_extent = {'west': 90, 'east': 175, 'north': 30, 'south':-50} #??
cpac_download_extent = {'west': 132, 'east': 205, 'north': 12, 'south':-22} #central pacific

#choose which subregions to download, only keep global if you want to do the whole thing, separate takes more time but saves space
download_extents = [global_download_extent] 
extent_names = ['global']

##-180,180,90,-90 with 0,6,12,18 for last 14 days available for a given month results in 6.49GB file timing out - months pulled toghet
## what if we pulled the months separately?

 