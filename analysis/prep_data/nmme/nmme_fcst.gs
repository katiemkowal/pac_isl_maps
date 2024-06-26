'reinit'
'open nmme_sst_ensmean_fcst.ctl'
'set lat -90 90'
'set lon -180 180'
zz = 3 + 1
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/nmme/dat_files/nmme_threeseas_fcst_sst_ld_3.dat'

'define tt = ave(fcst,z='zz+0',z='zz+2')*60*60*24'
'd re(tt,360,linear,-180,1.0,181,linear,-90,1.0,ba)'
*'d tt'
'disable fwrite'
'quit'
