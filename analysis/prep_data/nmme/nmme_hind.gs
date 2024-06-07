'reinit'
'open mayic_ENSM_MEAN_1991-2022.ctl'
'open junic_ENSM_MEAN_1991-2022.ctl'
'open julic_ENSM_MEAN_1991-2022.ctl'
'set lat -90 90'
'set lon -180 180'
zz = 3 + 1 
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/nmme/dat_files/nmmec_hind_precip_ld_3.dat'
i=8
while(i<=32)
'set t 'i
'set dfile 1'
'define tt = ave(fcst.1,z='zz+0',z='zz+2')'
'd tt' 
'set dfile 2'
'define tt = ave(fcst.2,z='zz+0',z='zz+2')'
'd tt'
'set dfile 3'
'define tt = ave(fcst.3,z='zz+0',z='zz+2')'
'd tt'
i = i + 1
endwhile
'disable fwrite'
'quit'
