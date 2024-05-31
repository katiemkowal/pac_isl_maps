'reinit'
'open grid.ctl'
'set y 1 201'
'set x 1 721'
'define grd=prc'
'close 1' 
'open globe_mask0p1.ctl'
'set lat -50 50'
'set lon -180 180'
'define mm = mask'
'close 1'
'set gxout fwrite'
'set fwrite chirps_ld3.dat'
i=1991
while(i<=2022)
'open chirps_daily.ctl'
'set lat -50 50'
'set lon -180 180'
yr = i
yrr = i
say Aug
say Oct
say 92
if(Aug=Nov & Oct=Jan);yr=i;yrr=i+1;endif
if(Aug=Dec & Oct=Feb);yr=i;yrr=i+1;endif
if(Aug=Jan & Oct=Mar);yr=i+1;yrr=i+1;endif
if(Aug=Feb & Oct=Apr);yr=i+1;yrr=i+1;endif
if(Aug=Mar & Oct=May);yr=i+1;yrr=i+1;endif

say yr ' ' yrr

'define tt = ave(precip,time=1Aug'yr',time=31Oct'yrr')'
'define ttt = maskout(lterp(tt,mm),mm)'
'd re(ttt,721,linear,-180,0.5,201,linear,-50,0.5,ba)'
i = i + 1
endwhile
'disable fwrite'
'quit'
