# configuration file for XCast Seasonal Data for Pacific Islands

#directories
nmme_dir = '/cpc/int_desk/pac_isl/data/processed/nmme/nc_files'
chirps_dir = '/cpc/int_desk/pac_isl/data/processed/chirps/nc_files'


#conditions used to set dry mask (don't think we need this if pac islands is so wet?)
dry_threshold = 0.2 #abosolute vale criteria in mm for no rainfall
quantile_threshold = 0.5 #percentage of grid cells required to fall below dry_threshold to be masked as dry

#number of leads to pull; lead 1 will be a seasonal forecast starting the following month, e.g. Model Initialized in May is MJJ 
leads = [1,2,3]

#hindcast years
hstart = 1991
hend = 2022

#cca training modes (letting xcast automatically search takes longer)
x_modes = 10
y_modes = 10
cca_modes = 8

#training windows
lov_window = 1 #number of years to leave out in cross validation analysis

#predictor_zone (large zone upon which to train models)
predictor_w=120
predictor_e=250
predictor_s=-30
predictor_n=0

#predictand zone (target zone)
predictand_w=177
predictand_e=183
predictand_s=-20
predictand_n=-15