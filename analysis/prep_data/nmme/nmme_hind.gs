'reinit'
'open junic_ENSM_MEAN_1991-2022.ctl'
'set lat -90 90'
'set lon -180 180'
zz = 3 + 1 
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/nmme/dat_files/nmme_hind_precip_ld_3.dat'
i=8
while(i<=32)
'set t 'i
#'set dfile 1'
'define tt = ave(fcst,z='zz+0',z='zz+2')'
'd tt' 
'd re(tt,360,linear,-180,1.0,181,linear,-90,1.0,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
