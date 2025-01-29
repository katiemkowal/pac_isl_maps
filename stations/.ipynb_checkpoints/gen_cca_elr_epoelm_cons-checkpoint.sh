cd /cpc/africawrf/ebekele/projects/PREPARE_pacific/notebooks
dtt=`date --date "0 day ago" "+%Y%m%d"`
dt=`date --date "0 day ago" "+%d%b%Y"`
iwk1=`date --date "-1 day ago" "+%d%b%Y"`
iwk2=`date --date "-8 day ago" "+%d%b%Y"`
iwk3=`date --date "-15 day ago" "+%d%b%Y"`
iwk34=`date --date "-15 day ago" "+%d%b%Y"`
iwk1234=`date --date "-1 day ago" "+%d%b%Y"`

fwk1=`date --date "-7 day ago" "+%d%b%Y"`
fwk2=`date --date "-14 day ago" "+%d%b%Y"`
fwk3=`date --date "-21 day ago" "+%d%b%Y"`
fwk34=`date --date "-28 day ago" "+%d%b%Y"`
fwk1234=`date --date "-28 day ago" "+%d%b%Y"`

for wk in $1 ; do

if [ $wk = 1 ]; then iwk=$iwk1; fwk=$fwk1; fi
if [ $wk = 2 ]; then iwk=$iwk2; fwk=$fwk2; fi
if [ $wk = 3 ]; then iwk=$iwk3; fwk=$fwk3; fi
if [ $wk = 34 ]; then iwk=$iwk34; fwk=$fwk34; fi
if [ $wk = 1234 ]; then iwk=$iwk1234; fwk=$fwk1234; fi

mmndy=`date +"%m"-"%d"`
mondy=`date +'%-m, %-d'`
mndy=`date -d "1 day ago" "+%m%d"`
yr=`date -d "1 day ago" "+%Y"`
mn=`date -d "1 day ago" "+%m"`
dy=`date -d "1 day ago" "+%d"`

wkn=`date '+%W'`

cat>gen_tercile_cons.gs<<eofGS
'reinit'
'open cca_bnn_${wk}.ctl'
'open cca_nnn_${wk}.ctl'
'open cca_ann_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define bncca = prob.1'
'define nncca = prob.2'
'define ancca = prob.3'
'close 3'
'close 2'
'close 1'
'open elr_bnn_${wk}.ctl'
'open elr_nnn_${wk}.ctl'
'open elr_ann_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define bnelr = prob.1'
'define nnelr = prob.2'
'define anelr = prob.3'
'close 3'
'close 2'
'close 1'
'open epoelm_bnn_${wk}.ctl'
'open epoelm_nnn_${wk}.ctl'
'open epoelm_ann_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define bnepoelm = prob.1'
'define nnepoelm = prob.2'
'define anepoelm = prob.3'
'close 3'
'close 2'
'close 1'
'open cca_rpss_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define rcca = rr'
'close 1'
'open elr_rpss_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define relr = rr'
'close 1'
'open epoelm_rpss_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define repoelm = rr'

'define rr1 = const(const(maskout(rcca,rcca-0.01),1),0,-u)'
'define rr2 = const(const(maskout(relr,relr-0.01),1),0,-u)'
'define rr3 = const(const(maskout(repoelm,repoelm-0.01),1),0,-u)'

'define scca = rr1*rcca'
'define selr = rr2*relr'
'define sepoelm = rr3*repoelm'

'define consan = ((scca*ancca) + (selr*anelr) + (sepoelm*anepoelm)) /(scca + selr + sepoelm)'
'define consnn = ((scca*nncca) + (selr*nnelr) + (sepoelm*nnepoelm)) /(scca + selr + sepoelm)'
'define consbn = ((scca*bncca) + (selr*bnelr) + (sepoelm*bnepoelm)) /(scca + selr + sepoelm)'

'define pconsan = consan /(consan+consnn+consbn)'
'define pconsnn = consnn /(consan+consnn+consbn)'
'define pconsbn = consbn /(consan+consnn+consbn)'
'close 1'
'open /cpc/home/ebekele/gen_mask_for_grads/pacific.ctl'
'set lat -20 3'
'set lon 155 181'
'define mm = mask'
'define pan = maskout(lterp(pconsan,mm),mm)*100'
'define pnn = maskout(lterp(pconsnn,mm),mm)*100'
'define pbn = maskout(lterp(pconsbn,mm),mm)*100'
'define an = maskout(maskout(pan,pan-pnn),pan-pbn)'
'define nn = maskout(maskout(pnn,pnn-pan),pnn-pbn)'
'define bn = maskout(maskout(pbn,pbn-pan),pbn-pnn)'

'define avanchk = aave(an,global)'
'define avnnchk = aave(nn,global)'
'define avbnchk = aave(bn,global)'

'd avanchk'; res1 = sublin(result,1); undan = subwrd(res1,4)
'd avnnchk'; res2 = sublin(result,1); undnn = subwrd(res2,4)
'd avbnchk'; res3 = sublin(result,1); undbn = subwrd(res3,4)

'set gxout grfill'
'set mpdraw off'
'set rgb 21 0 78 68 '
'set rgb 22 1 96 88 '
'set rgb 23 16 116 108 '
'set rgb 24 39 138 130 '
'set rgb 25 65 159 151 '
'set rgb 26 97 183 172 '
'set rgb 27 129 206 194 '
'set rgb 28 163 219 211 '
'set rgb 29 193 232 226 '
'set rgb 30 215 238 235 '
'set rgb 31 235 243 242 '
'set rgb 32 245 242 234 '
'set rgb 33 246 237 213 '
'set rgb 34 244 229 190 '
'set rgb 35 234 213 159 '
'set rgb 36 223 195 126 '
'set rgb 37 210 167 92 '
'set rgb 38 196 139 58 '
'set rgb 39 177 116 35 '
'set rgb 40 155 95 20 '
'set rgb 41 132 76 9 '
'set rgb 42 108 62 7 '
'set rgb 43 245 245 245 '
'set rgb 44 233 233 233 '
'set rgb 45 217 217 217 '
'set rgb 46 198 198 198 '
'set rgb 47 176 176 176 '
'set rgb 48 149 149 149 '
'set rgb 49 126 126 126 '
'set rgb 50 104 104 104 '
'set rgb 51 81 81 81 '
'set rgb 52 51 51 51 '
'set rgb 53 24 24 24 '

'set display color white'
'c'
'set xlint 5'
'set ylint 5'
'set grads off'
if(undan>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 30 29 28 27 26 25 24 23 22 21'

'd an'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 6.35 8.85 0.25 0.45'
endif
if(undnn>0)
'set clevs 40 45 50 55'
'set ccols 46 47 48 48 49'
'd nn'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 4.5 6.2 0.25 0.45'
endif
if(undbn>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 33 34 35 36 37 38 39 40 41 42'
'd bn'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 1.85 4.35 0.25 0.45'
endif
'set line 1 1 6'
'draw shp /cpc/home/ebekele/packages/subseason/gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.11'
'draw string 5.5 7.9 GEFS, Week-${wk}, S. Weighted Cons., Valid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cons.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cons.png gefs_week_${wk}_cons.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cons.png gefs_week_${wk}_cons.png'
'quit'
eofGS

/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads -blc gen_tercile_cons.gs

done
