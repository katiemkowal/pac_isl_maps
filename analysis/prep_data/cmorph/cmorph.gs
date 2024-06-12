'reinit'
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/cmorph/dat_files/cmorph_ld1.dat'
i=1998
while(i<=2022)
'open cmorph.ctl'
'set lat -50 50'
'set lon -180 180'
yr = i
yrr = i
say Jan ' ' Feb ' ' Mar
say Mar ' ' Apr ' ' May
say 90 ' ' 89 ' ' 92
if(Jan=Nov & Mar=Jan);yr=i;yrr=i+1;endif
if(Jan=Dec & Mar=Feb);yr=i;yrr=i+1;endif
if(Jan=Jan & Mar=Mar);yr=i+1;yrr=i+1;endif
if(Jan=Feb & Mar=Apr);yr=i+1;yrr=i+1;endif
if(Jan=Mar & Mar=May);yr=i+1;yrr=i+1;endif

if(Feb=Nov & Apr=Jan);yr=i;yrr=i+1;endif
if(Feb=Dec & Apr=Feb);yr=i;yrr=i+1;endif
if(Feb=Jan & Apr=Mar);yr=i+1;yrr=i+1;endif
if(Feb=Feb & Apr=Apr);yr=i+1;yrr=i+1;endif
if(Feb=Mar & Apr=May);yr=i+1;yrr=i+1;endif

if(Mar=Nov & May=Jan);yr=i;yrr=i+1;endif
if(Mar=Dec & May=Feb);yr=i;yrr=i+1;endif
if(Mar=Jan & May=Mar);yr=i+1;yrr=i+1;endif
if(Mar=Feb & May=Apr);yr=i+1;yrr=i+1;endif
if(Mar=Mar & May=May);yr=i+1;yrr=i+1;endif

say yr ' ' yrr

'define tt = ave(r,time=1Jan'yr',time=Mar'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1Feb'yr',time=Apr'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1Mar'yr',time=May'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
