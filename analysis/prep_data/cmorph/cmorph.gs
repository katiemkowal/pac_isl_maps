'reinit'
'set gxout fwrite'
'set fwrite /cpc/int_desk/pac_isl/data/processed/cmorph/dat_files/cmorph_ld3.dat'
i=1998
while(i<=2022)
'open cmorph.ctl'
'set lat -50 50'
'set lon -180 180'
yr = i
yrr = i
say Aug ' ' Sep ' ' Oct
say Oct ' ' Nov ' ' Dec
say 92 ' ' 91 ' ' 92
if(Aug=Nov & Oct=Jan);yr=i;yrr=i+1;endif
if(Aug=Dec & Oct=Feb);yr=i;yrr=i+1;endif
if(Aug=Jan & Oct=Mar);yr=i+1;yrr=i+1;endif
if(Aug=Feb & Oct=Apr);yr=i+1;yrr=i+1;endif
if(Aug=Mar & Oct=May);yr=i+1;yrr=i+1;endif

if(Sep=Nov & Nov=Jan);yr=i;yrr=i+1;endif
if(Sep=Dec & Nov=Feb);yr=i;yrr=i+1;endif
if(Sep=Jan & Nov=Mar);yr=i+1;yrr=i+1;endif
if(Sep=Feb & Nov=Apr);yr=i+1;yrr=i+1;endif
if(Sep=Mar & Nov=May);yr=i+1;yrr=i+1;endif

if(Oct=Nov & Dec=Jan);yr=i;yrr=i+1;endif
if(Oct=Dec & Dec=Feb);yr=i;yrr=i+1;endif
if(Oct=Jan & Dec=Mar);yr=i+1;yrr=i+1;endif
if(Oct=Feb & Dec=Apr);yr=i+1;yrr=i+1;endif
if(Oct=Mar & Dec=May);yr=i+1;yrr=i+1;endif

say yr ' ' yrr

'define tt = ave(r,time=1Aug'yr',time=31Oct'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1Sep'yr',time=30Nov'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
'define tt = ave(r,time=1Oct'yr',time=31Dec'yrr')'
'd re(tt,1441,linear, -180,0.25,401,linear,-50,0.25,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
