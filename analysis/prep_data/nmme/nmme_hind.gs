'reinit'
'open ../globe_mask0p1.ctl'
'set lat -50 50'
'set lon -180 180'
'define mm = mask'
'close 1'
'open ../chirps/chirps_ld3.ctl'
'set lat -50 50'
'set lon -180 180'
'define clm = ave(precip,t=1,t=30)'
'define clmm = lterp(clm,mm)'
'close 1'
'open junic_ENSM_MEAN_1991-2022.ctl'
'set lat -90 90'
'set lon -180 180'
zz = 3 + 1 
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/nmme/dat_files/nmme_oneseas_hind_precip_ld_3.dat'
i=1
while(i<=32)
'set t 'i
'define tt = ave(fcst,z='zz+0',z='zz+2')'
'define ttt = lterp(tt,mm)'
'define ttm = maskout(maskout(ttt,(clmm-0.5)),mm)'
'd re(tt,360,linear,-180,1.0,181,linear,-90,1.0,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
