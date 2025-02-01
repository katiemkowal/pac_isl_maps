cd /cpc/africawrf/ebekele/projects/PREPARE_pacific/subseason_unmasked
#!/bin/bash
############################################################
# This Shell script and the embeded GrADS Script are used to
# Download GEFS Forecast Data for domain of your interest and
# Plots Week-1 Precip/ T2m and Circulation Anomaly Forecasts
# Written by Endalkachew Bekele, NOAA/CPC/International Desks, August 2022

LANG=en_US.utf8
LC_ALL=en_US.utf8
unameout="$(uname -s)"

   wk1=`date --date "-1 day ago" "+%d%b"`
   wkk2=`date --date "-1 day ago" "+%d%b%Y"`
   iwk1=`date --date "-1 day ago" "+%Y%m%d"`
   fwk1=`date --date "-7 day ago" "+%Y%m%d"`
   dtt1=`date --date "-1 day ago" "+%d%b%Y"`
   dtt2=`date --date "-2 day ago" "+%d%b%Y"`
   dtt3=`date --date "-3 day ago" "+%d%b%Y"`
   dtt4=`date --date "-4 day ago" "+%d%b%Y"`
   dtt5=`date --date "-5 day ago" "+%d%b%Y"`
   dtt6=`date --date "-6 day ago" "+%d%b%Y"`
   dtt7=`date --date "-7 day ago" "+%d%b%Y"`
   ciwk1=`date --date "-1 day ago" "+%d%b"`
   cfwk1=`date --date "-7 day ago" "+%d%b"`

dt=`date +%Y%m%d`
dtt=`date +%d%b%Y`

if [ "$#" -ne 4 ]; then
    echo "***************************************************************************************************************************"
    echo " error while running plot_gefs_week1_anomalies.sh"
    echo "----------------- You need to provide 4 arguments -------------------------------------------------------------------------"
    echo "***************************************************************************************************************************"
    echo
    echo "The four arguments reflect geographical extent of your area:"
    echo
    echo "western lon, eastern lon, southern lat and northern lat in degreas, respectively"
    echo
    echo "Use negative sign for longitudes in western, and latitudes in the southern hemisphere"
    echo
    echo "For example, if your area extends from 41E to 45E and from 22S to 45N .."
    echo " you may run the script as .."
    echo
    echo " ./plot_gefs_week1_plots.sh 41 45 -22 45"
    echo
    echo
exit
fi

# This sets current date for GEFS data Download

##### Define your domain here#####
west=$1
east=$2
south=$3
north=$4
########################333

# This extends the domain area by 1 degree for horizontal divergence calculation

let west2=$west-11
let east2=$east+11
let south2=$south-11
let north2=$north+11

let west1=$west2-1
let east1=$east2+1
let south1=$south2-1
let north1=$north2+1


# This calculates the number of grids in the zomnla and meridional directions
let dxx=$east1-$west1+2
let dyy=$north1-$south1+2


#gribmapdir=/cpc/home/ebekele/opg2/grads-2.0.1.oga.1/Contents

cat>gefs_week1_wind_${dt}IC.ctl<<eofctl
dset ^gefs_week1_wind_${dt}IC.dat
undef -999000000.000000
xdef 360 linear 0 1
ydef 181 linear -90 1
zdef 1 linear 1 1
tdef 1 linear ${dtt} 1dy
vars 20
u850mb 0 99 u wind 850mb
v850mb 0 99 v wind 850mb
u700mb 0 99 u wind 700mb
v700mb 0 99 v wind 700mb
u500mb 0 99 u wind 500mb
v500mb 0 99 v wind 500mb
u200mb 0 99 u wind 200mb
v200mb 0 99 v wind 200mb
mslp 0 99 u mean sea level Pressure
h500mb 0 99 500 height
u850mbt 0 99 u wind total 850mb
v850mbt 0 99 v wind total 850mb
u700mbt 0 99 u wind total 700mb
v700mbt 0 99 v wind total 700mb
u500mbt 0 99 u wind total 500mb
v500mbt 0 99 v wind total 500mb
u200mbt 0 99 u wind total 200mb
v200mbt 0 99 v wind total 200mb
mslpt 0 99 u mean sea level Pressure total
h500mbt 0 99 500 height total
endvars
eofctl

cat>gefs_week1_vpot_${dt}IC.ctl<<eofctl
dset ^gefs_week1_vpot_${dt}IC.dat
undef -999000000.000000
xdef 360 linear 0 1
ydef 181 linear -90 1
zdef 1 linear 1 1
tdef 1 linear ${dtt} 1dy
vars 6
vpot850 0 99 850mb VPot
udiv850 0 99 850 u Div wind
vdiv850 0 99 850 v Div Wind
vpot200 0 99 200mb VPot
udiv200 0 99 200 u Div wind
vdiv200 0 99 200 v Div Wind
endvars
eofctl


cat>gefs_week1_precip_${dt}IC.ctl<<eofctl
dset ^gefs_week1_precip_${dt}IC.dat
undef -999000000.000000
xdef 720 linear 0 0.5
ydef 361 linear -90 0.5
zdef 1 linear 1 1
tdef 1 linear ${dtt} 1dy
vars 15
clim 0 99 7-day climatology
total 0 99 7-day toatl Fcst
rawabv 0 99 Above Prob Raw fcst
calibabv 0 99 Above prob Calib fcst
prob10 0 99 Exccedance prob 10mm
prob25 0 99 Exccedance prob 25mm
prob50 0 99 Exccedance prob 50mm
prob100 0 99 Exccedance prob 100mm
prob150 0 99 Exccedance prob 150mm
prob200 0 99 Exccedance prob 200mm
prob20 0 99 Exccedance prob 200mm
prob30 0 99 Exccedance prob 200mm
prob40 0 99 Exccedance prob 200mm
prob75 0 99 Exccedance prob 200mm
corr 0 99 Correlation
endvars
eofctl

cat>gefs_week1_t2m_${dt}IC.ctl<<eofctl
dset ^gefs_week1_t2m_${dt}IC.dat
undef -999000000.000000
xdef 720 linear 0 0.5
ydef 361 linear -90 0.5
zdef 1 linear 1 1
tdef 1 linear ${dtt} 1dy
vars 5
clim 0 99 7-day climatology
total 0 99 7-day toatl Fcst
rawabv 0 99 Above Prob Raw fcst
calibabv 0 99 Above prob Calib fcst
corr 0 99 Correlation
endvars
eofctl

cat>gefs_realtime_week1_plots.gs<<eofGS
'reinit'
'open /cpc/home/ebekele/gen_mask_for_grads/pacific.ctl'
'set lat $south1 $north1'
'set lon $west1 $east1'
'define msk = mask'
'close 1'
* Calculates Horizontal Divergence
'open gefs_week1_wind_${dt}IC.ctl'
'set lat $south1 $north1'
'set lon $west1 $east1'
'define mslpwk1total = mslpt/100'
'define mslpwk1anom = mslp'
'define mslpwk1clim = mslpt/100-mslp'
'define h500wk1total = h500mbt'
'define h500wk1anom = h500mb'
'define h500wk1clim = h500mbt-h500mb'

L=1
while (L <= 4)
  if (L=1); lv=850; endif
  if (L=2); lv=700; endif
  if (L=3); lv=500; endif
  if (L=4); lv=200; endif
**
'set lev 'lv''
'define u'lv' = u'lv'mbt'
'define v'lv' = v'lv'mbt'
'define uc'lv' = u'lv'mbt-u'lv'mb'
'define vc'lv' = v'lv'mbt-v'lv'mb'
'define uanom'lv' = u'lv'mb'
'define vanom'lv' = v'lv'mb'

'define divganom'lv' = hdivg(uanom'lv',vanom'lv') * 100000'
'define vortanom'lv' = hcurl(uanom'lv',vanom'lv') * 100000'
L=L+1
endwhile
'close 1'

'open gefs_week1_vpot_${dt}IC.ctl'
'set lat $south1 $north1'
'set lon $west1 $east1'

L=1
while (L <= 2)
  if (L=1); lv=850; endif
  if (L=2); lv=200; endif
**
'define vpot'lv' = vpot'lv''
'define udiv'lv' = udiv'lv''
'define vdiv'lv' = vdiv'lv''
L = L + 1
endwhile
'close 1'

* Precipitation
'set mpdset hires'
'set map 1 1 6'
'gradssupp/define_colors'
'open gefs_week1_precip_${dt}IC.ctl'
'set lat $south1 $north1'
'set lon $west1 $east1'
* Raw Anomaly
'define prcwk1anomraw = total - clim'
* Raw, Probability of Above-Average
'define prcwk1abv = rawabv'
* Raw Probability of Below Average
'define prcwk1blw = 1 - prcwk1abv'
* Maskout for Raw Probability of Above-average
'define prcmaskabv2r = prcwk1abv - 0.50'
* Maskout for Raw Probability of below-average
'define prcmaskblw2r = prcwk1blw - 0.50'
* Calibrated, probability of above-average
'define prcprobabv2 = calibabv'
* Calibrated, probability of below-average
'define prcprobblw2 = 1 - prcprobabv2'
* Maskout for calibrated Probability of Above-average
'define prcmaskabv2c = prcprobabv2 - 0.50'
* Maskout for calibrated Probability of below-average
'define prcmaskblw2c = prcprobblw2 - 0.50'
* Probability of Exceedance
'define prcwk1total = total'
'define prob10wk1 = prob10'
'define prob25wk1 = prob25'
'define prob50wk1 = prob50'
'define prob100wk1 = prob100'
'define prob150wk1 = prob150'
'define prob200wk1 = prob200'
'define precipcorr=corr'
* Daily Climatology
'define dlyclimwk1=clim/7.0'
* Dry mask (daily rainfall < 0.5mm)
'define dmaskwk1=dlyclimwk1-0.5'
'define dmskbin = const(const(maskout(dmaskwk1,dmaskwk1),1),0,-u)'
'define dmskchk = asum(dmskbin,lon=$west,lon=$east,lat=$south,lat=$north)'
'd dmskchk';res1=sublin(result,1);res2=subwrd(res1,4)

'define pabvbinr = const(const(maskout(prcmaskabv2r,prcmaskabv2r),1),0,-u)'
'define pabvchkr = asum(pabvbinr,lon=$west,lon=$east,lat=$south,lat=$north)'
'd pabvchkr';res11=sublin(result,1);res3=subwrd(res11,4)

'define pblwbinr = const(const(maskout(prcmaskblw2r,prcmaskblw2r),1),0,-u)'
'define pblwchkr = asum(pblwbinr,lon=$west,lon=$east,lat=$south,lat=$north)'
'd pblwchkr';res22=sublin(result,1);res4=subwrd(res22,4)

'define pabvbinc = const(const(maskout(prcmaskabv2c,prcmaskabv2c),1),0,-u)'
'define pabvchkc = asum(pabvbinc,lon=$west,lon=$east,lat=$south,lat=$north)'
'd pabvchkc';res33=sublin(result,1);res5=subwrd(res33,4)

'define pblwbinc = const(const(maskout(prcmaskblw2c,prcmaskblw2c),1),0,-u)'
'define pblwchkc = asum(pblwbinc,lon=$west,lon=$east,lat=$south,lat=$north)'
'd pblwchkc';res44=sublin(result,1);res6=subwrd(res44,4)

'close 1'

* 2m temperature
'open gefs_week1_t2m_${dt}IC.ctl'
'set lat $south1 $north1'
'set lon $west1 $east1'
* Raw Anomaly
'define t2mwk1anomraw = total - clim'
* Raw, Probability of Above-Average
'define t2mwk1abv = rawabv'
* Raw Probability of Below Average
'define t2mwk1blw = 1 - t2mwk1abv'
* Maskout for Raw Probability of Above-average
'define t2mmaskabv2r = t2mwk1abv - 0.50'
* Maskout for Raw Probability of below-average
'define t2mmaskblw2r = t2mwk1blw - 0.50'
* Calibrated, probability of above-average
'define t2mprobabv2 = calibabv'
* Calibrated, probability of below-average
'define t2mprobblw2 = 1 - t2mprobabv2'
* Maskout for calibrated Probability of Above-average
'define t2mmaskabv2c = t2mprobabv2 - 0.50'
* Maskout for calibrated Probability of below-average
'define t2mmaskblw2c = t2mprobblw2 - 0.50'
'define t2mwk1total = total'
'define t2mcorr=corr'

'define tabvbinr = const(const(maskout(t2mmaskabv2r,t2mmaskabv2r),1),0,-u)'
'define tabvchkr = asum(tabvbinr,lon=$west,lon=$east,lat=$south,lat=$north)'
'd tabvchkr';res55=sublin(result,1);res7=subwrd(res55,4)

'define tblwbinr = const(const(maskout(t2mmaskblw2r,t2mmaskblw2r),1),0,-u)'
'define tblwchkr = asum(tblwbinr,lon=$west,lon=$east,lat=$south,lat=$north)'
'd tblwchkr';res66=sublin(result,1);res8=subwrd(res66,4)

'define tabvbinc = const(const(maskout(t2mmaskabv2c,t2mmaskabv2c),1),0,-u)'
'define tabvchkc = asum(tabvbinc,lon=$west,lon=$east,lat=$south,lat=$north)'
'd tabvchkc';res77=sublin(result,1);res9=subwrd(res77,4)


'define tblwbinc = const(const(maskout(t2mmaskblw2c,t2mmaskblw2c),1),0,-u)'
'define tblwchkc = asum(tblwbinc,lon=$west,lon=$east,lat=$south,lat=$north)'
'd tblwchkc';res88=sublin(result,1);res10=subwrd(res88,4)

'set gxout shaded'
'set display color white'
'c'
'set gxout shaded'
if($dxx>80);xl=15;xskp=3;endif
if($dxx<=80&$dxx>40);xl=10;xskp=2;endif
if($dxx<=40&$dxx>20);xl=5;xskp=1;endif
if($dxx<=20&$dxx>10);xl=3;xskp=1;endif
if($dxx<=10&$dxx>5);xl=2;xskp=1;endif
if($dxx<5);xl=1;xskp=13;endif

if($dyy>80);yl=15;yskp=3;endif
if($dyy<=80&$dyy>40);yl=10;yskp=2;endif
if($dyy<=40&$dyy>20);yl=5;yskp=1;endif
if($dyy<=20&$dyy>10);yl=3;yskp=1;endif
if($dyy<=10&$dyy>5);yl=2;yskp=1;endif
if($dyy<5);yl=1;yskp=1;endif


* Week-1, MSLP Wind Total
'set lat $south2 $north2'
'set lon $west2 $east2'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 996 1000 1004 1008 1012 1016 1020 1024 1028 1032 1036'
'set ccols 29 27 25 23 22 21 41 42 43 45 47 49'
'd smth9(mslpwk1total)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs 996 1000 1004 1008 1012 1016 1020 1024 1028 1032 1036'
'set ccols 9 9 9 9 9 9 9 9 9 9 9 9 9 9'
'd smth9(mslpwk1total)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Mean Sea Level Pressure Total'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_mslpt.png'
'!convert -trim gefs_week1_mslpt.png gefs_week1_mslpt.png'
'!convert -bordercolor white -border 10 gefs_week1_mslpt.png gefs_week1_mslpt.png'
'c'
* Week-1, MSLP Wind Climo
'set lat $south2 $north2'
'set lon $west2 $east2'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 996 1000 1004 1008 1012 1016 1020 1024 1028 1032 1036'
'set ccols 29 27 25 23 22 21 41 42 43 45 47 49'
'd smth9(mslpwk1clim)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs 996 1000 1004 1008 1012 1016 1020 1024 1028 1032 1036'
'set ccols 9 9 9 9 9 9 9 9 9 9 9 9 9 9'
'd smth9(mslpwk1clim)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Mean Sea Level Pressure Climatology'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${ciwk1} - ${cfwk1}'
'printim gefs_week1_mslpc.png'
'!convert -trim gefs_week1_mslpc.png gefs_week1_mslpc.png'
'!convert -bordercolor white -border 10 gefs_week1_mslpc.png gefs_week1_mslpc.png'
'c'

* Week-1, MSLP Wind Anomaly
'set lat $south2 $north2'
'set lon $west2 $east2'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -4 -3 -2 -1 -0.5 0.5 1 2 3 4'
'set ccols 29 27 25 23 21 0 41 43 45 47 49'
'd smth9(mslpwk1anom)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs -4 -3 -2 -1 -0.5 0.5 1 2 3 4'
'set ccols 9 9 9 9 9 9 9 9 9 9 9'
'd smth9(mslpwk1anom)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Mean Sea Level Pressure Anomaly'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_mslp.png'
'!convert -trim gefs_week1_mslp.png gefs_week1_mslp.png'
'!convert -bordercolor white -border 10 gefs_week1_mslp.png gefs_week1_mslp.png'
'c'

L=1
while (L <= 4)
  if (L=1); lv=850; endif
  if (L=2); lv=700; endif
  if (L=3); lv=500; endif
  if (L=4); lv=200; endif
  
*************************
*** WIND TOTAL
*************************
'set lat $south2 $north2'
'set lon $west2 $east2'
'set mpdset hires'
'set map 1 1 6'
'gradssupp/define_colors'
'set display color white'
'c'
'set gxout shaded'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 1 2 3 4 5 6 8 10 15 20 30 40 50'
'set ccols 0 41 42 43 44 31 32 33 34 21 23 25 27 29'
'd mag(u'lv',v'lv')'
'gradssupp/cbarmerc2.gs'
'set gxout vector'
'set arrlab off'
'set cthick 12'
 'set arrscl 0.40 8.0'
 'set arrowhead -.5'
 'set ccolor 1'
if (L=4);'d skip(u'lv',5,5);v'lv'';endif
if (L<4);'d skip(u'lv',5,5);v'lv'';endif
*
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 'lv'-hPa Wind Total'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'set string 1 l 5'
'q w2xy $west2 $south2'; y2=subwrd(result,6)
'printim gefs_week1_'lv'_wind_spdt.png'
'!convert -trim gefs_week1_'lv'_wind_spdt.png gefs_week1_'lv'_wind_spdt.png'
'!convert -bordercolor white -border 10 gefs_week1_'lv'_wind_spdt.png gefs_week1_'lv'_wind_spdt.png'
'c'
* Wind Climo
'set lat $south2 $north2'
'set lon $west2 $east2'
'set mpdset hires'
'set map 1 1 6'
'gradssupp/define_colors'
'set display color white'
'c'
'set gxout shaded'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 1 2 3 4 5 6 8 10 15 20 30 40 50'
'set ccols 0 41 42 43 44 31 32 33 34 21 23 25 27 29'
'd mag(uc'lv',vc'lv')'
'gradssupp/cbarmerc2.gs'
'set gxout vector'
'set arrlab off'
'set cthick 12'
 'set arrscl 0.40 8.0'
 'set arrowhead -.5'
 'set ccolor 1'
if (L=4);'d skip(uc'lv',5,5);vc'lv'';endif
if (L<4);'d skip(uc'lv',5,5);vc'lv'';endif
*
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 'lv'-hPa Wind Climatology'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'set string 1 l 5'
'q w2xy $west2 $south2'; y2=subwrd(result,6)
'printim gefs_week1_'lv'_wind_spdc.png'
'!convert -trim gefs_week1_'lv'_wind_spdc.png gefs_week1_'lv'_wind_spdc.png'
'!convert -bordercolor white -border 10 gefs_week1_'lv'_wind_spdc.png gefs_week1_'lv'_wind_spdc.png'
'c'
'set gxout shaded'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -1.5 -1 -0.5 -0.25  0.25 0.5 1 1.5'
'set ccols  44 42 37 34  0 73 76 21 24 '
if (L=4);
'set ccols  24 21 76 73  0 34 37 42 44'
endif
'd divganom'lv''
'gradssupp/cbarmerc2.gs'
'set gxout vector'
'set arrlab off'
'set cthick 12'
 'set arrscl 0.40 8.0'
 'set arrowhead -.5'
 'set ccolor 1'
if (L=4);'d skip(uanom'lv',5,5);vanom'lv'';endif
if (L<4);'d skip(uanom'lv',5,5);vanom'lv'';endif
*
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 'lv'-hPa Div. and Wind Anom.'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'set string 1 l 5'
'q w2xy $west2 $south2'; y2=subwrd(result,6)
yy3 = y2 - 0.8
'set string 1 l 5'
'draw string 1.9 'yy3' Convergence'
'set string 1 l 5'
'draw string 5.4 'yy3' Divergence'
'printim gefs_week1_'lv'_wind_div.png'
'!convert -trim gefs_week1_'lv'_wind_div.png gefs_week1_'lv'_wind_div.png'
'!convert -bordercolor white -border 10 gefs_week1_'lv'_wind_div.png gefs_week1_'lv'_wind_div.png'
'c'
L = L + 1
endwhile
'c'

L=1
while (L <= 2)
  if (L=1); lv=850; endif
  if (L=2); lv=200; endif


'set gxout shaded'
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -4 -3 -2 -1 -0.5 0.5 1 2 3 4'
'set ccols  79 77 75 73 71 0 31 33 35 37 39'
if (L=2);
'set ccols  39 37 35 33 31 0 71 73 75 77 79'
endif
'd vpot'lv'*1e-06'
'gradssupp/cbarmerc2.gs'
'set gxout vector'
'set arrlab off'
'set cthick 12'
 'set arrscl 0.40 8.0'
 'set arrowhead -.5'
 'set ccolor 1'
if (L=2);'d skip(udiv'lv',5,5);vdiv'lv'';endif
if (L<2);'d skip(udiv'lv',5,5);vdiv'lv'';endif
*
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 'lv'-hPa VPOT and Divergent Wind Anom.'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'set string 1 l 5'
'q w2xy $west2 $south2'; y2=subwrd(result,6)
yy3 = y2 - 0.8
'set string 1 l 5'
'draw string 1.9 'yy3' Divergence'
'set string 1 l 5'
'draw string 5.4 'yy3' Convergence'
'printim gefs_week1_'lv'_vpot_div.png'
'!convert -trim gefs_week1_'lv'_vpot_div.png gefs_week1_'lv'_vpot_div.png'
'!convert -bordercolor white -border 10 gefs_week1_'lv'_vpot_div.png gefs_week1_'lv'_vpot_div.png'
'c'
L = L + 1
endwhile

* Week-1, 500-hPa Height Total
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5440 5480 5520 5560 5600 5640 5680 5720 5760 5800 5840 5880'
'set ccols 55 53 49 47 45 43 42 41 21 22 23 25 27 29'
'd smth9(h500wk1total)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs 5440 5480 5520 5560 5600 5640 5680 5720 5760 5800 5840 5880'
'set ccols 9 9 9 9 9 9 9 9 9 9 9 9 9 9'
'd smth9(h500wk1total)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 500-hPa Geo-Potential Height Total'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_500_heightt.png'
'!convert -trim gefs_week1_500_heightt.png gefs_week1_500_heightt.png'
'!convert -bordercolor white -border 10 gefs_week1_500_heightt.png gefs_week1_500_heightt.png'
'c'
* Week-1, 500-hPa Height Climo
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5440 5480 5520 5560 5600 5640 5680 5720 5760 5800 5840 5880'
'set ccols 55 53 49 47 45 43 42 41 21 22 23 25 27 29'
'd smth9(h500wk1clim)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs 5440 5480 5520 5560 5600 5640 5680 5720 5760 5800 5840 5880'
'set ccols 9 9 9 9 9 9 9 9 9 9 9 9 9 9'
'd smth9(h500wk1clim)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 500-hPa Geo-Potential Height Climatology'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${ciwk1} - ${cfwk1}'
'printim gefs_week1_500_heightc.png'
'!convert -trim gefs_week1_500_heightc.png gefs_week1_500_heightc.png'
'!convert -bordercolor white -border 10 gefs_week1_500_heightc.png gefs_week1_500_heightc.png'
'c'

* Week-1, 500-hPa Height Anomaly
'set grads off'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -20 -15 -10 -5 5 10 15 20'
'set ccols 48 46 44 42 0 22 24 26 28'
'd smth9(h500wk1anom)'
'set gxout contour'
'set cthick 6'
'set clab masked'
'set clevs -20 -15 -10 -5 5 10 15 20'
'set ccols 9 9 9 9 9 9 9 9 9'
'd smth9(h500wk1anom)'
'set gxout shaded'
'gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.15'
'q w2xy $west2 $north2'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 500-hPa Geo-Potential Height Anomaly'
'set strsiz 0.15'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_500_height.png'
'!convert -trim gefs_week1_500_height.png gefs_week1_500_height.png'
'!convert -bordercolor white -border 10 gefs_week1_500_height.png gefs_week1_500_height.png'
'c'
* Week-1, Rainfall Total
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 2 5 10 25 50 75 100 150 200 300 500 750 1000 1500 2500'
'set ccols 0 32 35 38 42 45 48 51 53 55 21 24 27 29 64 61'
'd lterp(prcwk1total,msk)' 
*'d maskout(lterp(prcwk1total,msk),msk)'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 Precip Total'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_precipt.png'
'!convert -trim gefs_week1_precipt.png gefs_week1_precipt.png'
'!convert -bordercolor white -border 10 gefs_week1_precipt.png gefs_week1_precipt.png'
'c'
* Week-1, Rainfall Anomaly
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -50 -40 -30 -20 -10 -5 5 10 20 30 40 50'
'set ccols  79 78 77 75 73 71 0 31 33 35 37 38 39'
'd lterp(prcwk1anomraw,msk)'
*'d maskout(lterp(prcwk1anomraw,msk),msk)'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 Precip Anomaly'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_precip.png'
'!convert -trim gefs_week1_precip.png gefs_week1_precip.png'
'!convert -bordercolor white -border 10 gefs_week1_precip.png gefs_week1_precip.png'
'set gxout grfill'
'c'
* Week-1, Rainfall Correlation
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'
'set ccols  49 48 47 46 45 44 43 42 41 0 21 22 23 24 25 26 27 28 29'
'd lterp(precipcorr,msk)'
*'d maskout(lterp(precipcorr,msk),msk)'
'gradssupp/cbarmerc2'
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 Precip Correlation'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${ciwk1} - ${cfwk1}'
'printim gefs_week1_precip_corr.png'
'!convert -trim gefs_week1_precip_corr.png gefs_week1_precip_corr.png'
'!convert -bordercolor white -border 10 gefs_week1_precip_corr.png gefs_week1_precip_corr.png'
'set gxout grfill'
'c'

* Week-1, Rainfall Raw
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/rgbset_brown2green2.gs'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
if(res3!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols  0 46 47 49 51 53'
'd maskout(lterp(prcwk1abv,msk),lterp(prcmaskabv2r,msk))'
*'d maskout(maskout(lterp(prcwk1abv,msk),msk),maskout(lterp(prcmaskabv2r,msk),msk))'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 4.3 7.80 'yyy1' 'yyy2''
endif
if(re4!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols 0 37 39 41 42 43'
'd maskout(lterp(prcwk1blw,msk),lterp(prcmaskblw2r,msk))'
*'d maskout(maskout(lterp(prcwk1blw,msk),msk),maskout(lterp(prcmaskblw2r,msk),msk))'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 0.50 4.00 'yyy1' 'yyy2''
endif
if(res2!=0)
'set rgb 91 235 235 235'
'set clevs 0'
'set ccols 91 91'
'd maskout(lterp(dlyclimwk1,msk),lterp(dmaskwk1,msk))'
*'d maskout(maskout(lterp(dlyclimwk1,msk),msk),maskout(lterp(dmaskwk1,msk),msk))'
endif
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 2-Category Precip Fcst.'
'set strsiz 0.16'
'draw string 4.25 'yy2' Raw, Valid: ${iwk1} - ${fwk1}'
'gradssupp/define_colors.gs'
'set string 1 l 5'
yy3 = yyy1 - 0.4
'set string 79 l 12'
'draw string 1.5 'yy3' Prob. of Below'
'set string 39 l 12'
'draw string 5.3 'yy3' Prob. of Above'
'printim gefs_week1_precip_raw_prob.png'
'!convert -trim gefs_week1_precip_raw_prob.png gefs_week1_precip_raw_prob.png'
'!convert -bordercolor white -border 10 gefs_week1_precip_raw_prob.png gefs_week1_precip_raw_prob.png'
'c'
* Week-1, Rainfall Reg. Calib.
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/rgbset_brown2green2.gs'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
if(res5!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols  0 46 47 49 51 53'
'd maskout(prcprobabv2,prcmaskabv2c)'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 4.3 7.80 'yyy1' 'yyy2''
endif
if(res6!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols 0 37 39 41 42 43'
'd maskout(prcprobblw2,prcmaskblw2c)'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 0.50 4.00 'yyy1' 'yyy2''
endif
if(res2!=0)
'set rgb 91 235 235 235'
'set clevs 0'
'set ccols 91 91'
'd maskout(lterp(dlyclimwk1,msk),lterp(dmaskwk1,msk))'
*'d maskout(maskout(lterp(dlyclimwk1,msk),msk),maskout(lterp(dmaskwk1,msk),msk))'
endif
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 2-Category Precip Forecast'
'set strsiz 0.16'
'draw string 4.25 'yy2' Calib., Valid: ${iwk1} - ${fwk1}'
'gradssupp/define_colors.gs'
'set string 1 l 5'
yy3 = yyy1 - 0.4
'set string 79 l 12'
'draw string 1.5 'yy3' Prob. of Below'
'set string 39 l 12'
'draw string 5.3 'yy3' Prob. of Above'
'printim gefs_week1_precip_calib_prob.png'
'!convert -trim gefs_week1_precip_calib_prob.png gefs_week1_precip_calib_prob.png'
'!convert -bordercolor white -border 10 gefs_week1_precip_calib_prob.png gefs_week1_precip_calib_prob.png'
'c'
'set gxout shaded'
* Week-1, Rainfall Exceedance Probability (>10mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob10wk1,msk)*100'
*'d maskout(lterp(prob10wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 10mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_10.png'
'!convert -trim gefs_week1_prob_10.png gefs_week1_prob_10.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_10.png gefs_week1_prob_10.png'
'c'

* Week-1, Rainfall Exceedance Probability (>25mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob25wk1,msk)*100'
*'d maskout(lterp(prob25wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 25mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_25.png'
'!convert -trim gefs_week1_prob_25.png gefs_week1_prob_25.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_25.png gefs_week1_prob_25.png'
'c'
* Week-1, Rainfall Exceedance Probability (>50mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob50wk1,msk)*100'
*'d maskout(lterp(prob50wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 50mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_50.png'
'!convert -trim gefs_week1_prob_50.png gefs_week1_prob_50.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_50.png gefs_week1_prob_50.png'
'c'
* Week-1, Rainfall Exceedance Probability (>100mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob100wk1,msk)*100'
*'d maskout(lterp(prob100wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 100mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_100.png'
'!convert -trim gefs_week1_prob_100.png gefs_week1_prob_100.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_100.png gefs_week1_prob_100.png'
'c'
* Week-1, Rainfall Exceedance Probability (>150mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob150wk1,msk)*100'
*'d maskout(lterp(prob150wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 150mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_150.png'
'!convert -trim gefs_week1_prob_150.png gefs_week1_prob_150.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_150.png gefs_week1_prob_150.png'
'c'
* Week-1, Rainfall Exceedance Probability (>200mm)
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 5 10 20 30 40 50 60 70 80 90 95'
'set ccols 0 31 33 35 37 39 43 52 54 56 58 48 49'
'd lterp(prob200wk1,msk)*100'
*'d maskout(lterp(prob200wk1,msk),msk)*100'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.18'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.3
'draw string 4.25 'yy1' GEFS Week-1 Exceedance Prob. > 200mm'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_prob_200.png'
'!convert -trim gefs_week1_prob_200.png gefs_week1_prob_200.png'
'!convert -bordercolor white -border 10 gefs_week1_prob_200.png gefs_week1_prob_200.png'
'c'
* Week-1, Temperature Total
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs 25 26 27 28 29 30 31 32 33 34'
'set ccols 34 33 32 21 23 25 27 29 65 64 63'
'd lterp(t2mwk1total,msk)'
*'d maskout(lterp(t2mwk1total,msk),msk)'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 T2m Total'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_t2mt.png'
'!convert -trim gefs_week1_t2mt.png gefs_week1_t2mt.png'
'!convert -bordercolor white -border 10 gefs_week1_t2mt.png gefs_week1_t2mt.png'
'c'
* Week-1, Temperature Correlation
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -0.9 -0.8 -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'
'set ccols  49 48 47 46 45 44 43 42 41 0 21 22 23 24 25 26 27 28 29'
'd t2mcorr'
'gradssupp/cbarmerc2'
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 T2m Correlation'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${ciwk1} - ${cfwk1}'
'printim gefs_week1_t2m_corr.png'
'!convert -trim gefs_week1_t2m_corr.png gefs_week1_t2m_corr.png'
'!convert -bordercolor white -border 10 gefs_week1_t2m_corr.png gefs_week1_t2m_corr.png'
'c'

* Week-1, Temperature Anomaly
'set grads off'
'gradssupp/define_colors'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
'set clevs -3.0 -2.0 -1.5 -1.0 -0.5 0.5 1.0 1.5 2.0 3.0'
'set ccols  49 47 45 43 41 0 21 23 25 27 29'
'd lterp(t2mwk1anomraw,msk)'
*'d maskout(lterp(t2mwk1anomraw,msk),msk)'
'gradssupp/cbarmerc2'
*'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 T2m Anomaly'
'set strsiz 0.16'
'draw string 4.25 'yy2' Valid: ${iwk1} - ${fwk1}'
'printim gefs_week1_t2m.png'
'!convert -trim gefs_week1_t2m.png gefs_week1_t2m.png'
'!convert -bordercolor white -border 10 gefs_week1_t2m.png gefs_week1_t2m.png'
'c'
'set gxout grfill'
* Week-1, Temperature Raw
'set grads off'
'gradssupp/rgbset_blue2red2.gs'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
if(res7!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols  0 21 22 23 24 25'
'd maskout(lterp(t2mwk1abv,msk),lterp(t2mmaskabv2r,msk))'
*'d maskout(maskout(lterp(t2mwk1abv,msk),msk),maskout(lterp(t2mmaskabv2r,msk),msk))'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 4.3 7.80 'yyy1' 'yyy2''
endif
if(res8!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols 0 20 19 18 17 16'
'd maskout(lterp(t2mwk1blw,msk),lterp(t2mmaskblw2r,msk))'
*'d maskout(maskout(lterp(t2mwk1blw,msk),msk),maskout(lterp(t2mmaskblw2r,msk),msk))'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 0.50 4.00 'yyy1' 'yyy2''
endif
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 2-Category T2m Fcst.'
'set strsiz 0.16'
'draw string 4.25 'yy2' Raw, Valid: ${iwk1} - ${fwk1}'
'gradssupp/define_colors.gs'
'set string 1 l 5'
yy3 = yyy1 - 0.4
'set string 49 l 12'
'draw string 1.5 'yy3' Prob. of Below'
'set string 29 l 12'
'draw string 5.3 'yy3' Prob. of Above'
'printim gefs_week1_t2m_raw_prob.png'
'!convert -trim gefs_week1_t2m_raw_prob.png gefs_week1_t2m_raw_prob.png'
'!convert -bordercolor white -border 10 gefs_week1_t2m_raw_prob.png gefs_week1_t2m_raw_prob.png'
'c'
* Week-1, T2m Calibrated
'set lat $south $north'
'set lon $west $east'
'set grads off'
'gradssupp/rgbset_blue2red2.gs'
'set xlint 'xl''
'set ylint 'yl''
'set xlopts 1 6 0.15'
'set ylopts 1 6 0.15'
if(res9!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols  0 21 22 23 24 25'
'd maskout(t2mprobabv2,t2mmaskabv2c)'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 4.3 7.80 'yyy1' 'yyy2''
endif
if(res10!=0)
'set clevs 0.50 0.55 0.60 0.70 0.80'
'set ccols 0 20 19 18 17 16'
'd maskout(t2mprobblw2,t2mmaskblw2c)'
'q w2xy $west $south'; y2=subwrd(result,6)
yyy2 = y2 - 0.3
yyy1 = y2 - 0.5
'gradssupp/xbar 0.50 4.00 'yyy1' 'yyy2''
endif
'set rgb 100 225 255 255'
*'gradssupp/basemap O 100 1 M'
'set line 1 1 6'
'draw shp gradssupp/WMO_basemap.shp'
'q w2xy $west $north'; y1=subwrd(result,6)
yy1 = y1 + 0.6
yy2 = y1 + 0.25
'set string 1 c'
'set strsiz 0.18'
'draw string 4.25 'yy1' GEFS Week-1 2-Category T2m Fcst.'
'set strsiz 0.16'
'draw string 4.25 'yy2' 	Calib., Valid: ${iwk1} - ${fwk1}'
'gradssupp/define_colors.gs'
'set string 1 l 5'
yy3 = yyy1 - 0.4
'set string 49 l 12'
'draw string 1.5 'yy3' Prob. of Below'
'set string 29 l 12'
'draw string 5.3 'yy3' Prob. of Above'
'printim gefs_week1_t2m_calib_prob.png'
'!convert -trim gefs_week1_t2m_calib_prob.png gefs_week1_t2m_calib_prob.png'
'!convert -bordercolor white -border 10 gefs_week1_t2m_calib_prob.png gefs_week1_t2m_calib_prob.png'
'c'
'set lat $south $north'
'set lon $west $east'
'set geotiff gefs_week1_precip_total.tif'
'set gxout geotiff'
'd prcwk1total'
'c'
'set geotiff gefs_week1_precip_anom.tif'
'd prcwk1anomraw'
'set gxout shaded'
'quit'
eofGS

# Runs The GrADS Script
/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads -bpc gefs_realtime_week1_plots.gs

# Moves all generated images into figures directory 
mv *.png gefs_week1_figures/
mv *.tif gefs_week1_figures/

