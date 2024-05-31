'reinit'
'open chirps_ld3.ctl'
'set y 1 201'
'set x 1 721'
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/chirps/dat_files/chirps_hind_precip_ld3.dat'
'define clm = ave(precip,t=1,t=30)'
i=1
while(i<=32)
'set t 'i
'define pp = maskout(precip,(clm-0.5))' 
'd re(pp,721,linear,-180,0.5,201,linear,-50,0.5,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
