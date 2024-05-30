# config file with constants for xcast analysis

#hindcast_dir
hcst_dir = '/cpc/africawrf/ebekele/cca/xcast/subseason/prep_data/data/'
#fcst_dir
fcst_dir = '/cpc/africawrf/ebekele/cca/xcast/subseason/prep_data/data/'
#observed_dir (to train model)
obs_dir = '/cpc/africawrf/ebekele/cca/xcast/subseason/prep_data/data/'

#temp_files folder
tmp_path = '/cpc/int_desk/pac_isl/analysis/xcast/'
#output_files folder
output_path = '/cpc/int_desk/pac_isl/output/ens_cons'

#conditions used to set dry mask
dry_threshold = 0.2 #abosolute vale criteria in mm for no rainfall
quantile_threshold = 0.5 #percentage of grid cells required to fall below dry_threshold to be masked as dry

#cca training modes (letting xcast automatically search takes longer)
x_modes = 5
y_modes = 5
cca_modes = 3

#training windows
lov_window = 5 #number of years to leave out in cross validation analysis