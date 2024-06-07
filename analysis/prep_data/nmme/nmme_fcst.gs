'reinit'
'open /cpc/fews/production/NMME/inputs/filtered/nmme_prate_ensmean_fcst.ctl'
'set lat -90 90'
'set lon -180 180'
zz = 3 + 1
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/nmme/dat_files/nmme_fcst_precip_ld_3.dat'

'define tt = ave(fcst,z='zz+0',z='zz+2')*60*60*24'
'd tt'
'disable fwrite'
'quit'
