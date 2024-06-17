'reinit'
'open cmorph_ld3.ctl'
'set y 1 401'
'set x 1 1441'
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/cmorph/dat_files/cmorph_hind_precip_ld3.dat'
#'define clm = ave(precip,t=26,t=50)'
i=1
while(i<=75)
'set t 'i
'define pp = precip'
#'define pp = maskout(precip,(clm-0.5))' 
'd re(pp,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
