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

iwkk=$(date -d"$iwk + 0 day" +"%d%b")
fwkk=$(date -d"$fwk + 0 day" +"%d%b")

rm temp.nc
rm gen_cca.py

cat>gen_cca.py<<eofPY
#!/usr/bin/env python
# coding: utf-8

import xcast as xc 
import xarray as xr 
import cartopy.crs as ccrs 
import numpy as np
import matplotlib.pyplot as plt 
model1 = xr.open_dataset('../prep_data/data/gefs_week${wk}_hind.nc')
obs1 = xr.open_dataset('../prep_data/data//cmorph_week${wk}_hind.nc')
fmodel1 = xr.open_dataset('../prep_data/data/gefs_week${wk}_fcst.nc')

model = model1.precip.expand_dims({'M':[0]})
fmodel = fmodel1.precip.expand_dims({'M':[0]})
obs = obs1.precip.expand_dims({'M':[0]})
#drymask = xc.drymask(obs, dry_threshold=0.2, quantile_threshold=0.5)
#obs = obs*drymask
#drymask = xc.drymask(model, dry_threshold=0.2, quantile_threshold=0.5)
#model = model*drymask

#drymask = xc.drymask(obs, dry_threshold=0.5, quantile_threshold=0.3)
#obs = obs*drymask

#obs, model = xc.match(obs, model)



mask_missing = model.mean('time', skipna=False).mean('M', skipna=False)
mask_missing = xr.ones_like(mask_missing).where(~np.isnan(mask_missing), other=np.nan )
model = model * mask_missing

model = model.sel(lon=slice(145, 191), lat=slice(-30,13))
fmodel = fmodel.sel(lon=slice(145, 191), lat=slice(-30,13))
obs = obs.sel(lon=slice(155, 181), lat=slice(-20,3))
model = xc.regrid(model, obs.lon, obs.lat)
fmodel = xc.regrid(fmodel, obs.lon, obs.lat)

hindcasts_prob = []
i=1
for xtrain, ytrain, xtest, ytest in xc.CrossValidator(model, obs, window=5):
    print("window {}".format(i))
    i += 1
    reg = xc.CCA(search_override=(5,5,3))
    reg.fit(xtrain, ytrain)
    probs =  reg.predict_proba(xtest)
#    probs = xc.gaussian_smooth(probs)
    hindcasts_prob.append(probs.isel(time=2))
hindcasts_prob = xr.concat(hindcasts_prob, 'time')

hindcasts_prob = xc.gaussian_smooth(hindcasts_prob, kernel=3)
obs = xc.gaussian_smooth(obs, kernel=3)

ohc = xc.OneHotEncoder() 
ohc.fit(obs)
T = ohc.transform(obs)
clim = xr.ones_like(T) * 0.333

fprobs =  reg.predict_proba(fmodel)
fprobs20 =  reg.predict_proba(fmodel,quantile=0.2)
fprobs80 =  1-(reg.predict_proba(fmodel,quantile=0.8))
fprobs10 =  reg.predict_proba(fmodel,quantile=0.1)
fprobs90 =  1-(reg.predict_proba(fmodel,quantile=0.9))


bnn = (fprobs[0,:,:,:])
nnn = (fprobs[1,:,:,:])
ann = (fprobs[2,:,:,:])

fout=np.float32(bnn)
fid=open("./cca_bnn_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(nnn)
fid=open("./cca_nnn_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(ann)
fid=open("./cca_ann_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

groc = xc.GROCS(hindcasts_prob, T)

fout=np.float32(groc)
fid=open("./cca_groc_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

clim_rps = xc.RankProbabilityScore(clim, T)
pred_rps = xc.RankProbabilityScore(hindcasts_prob, T)
rpss = 1 - pred_rps / clim_rps
fout=np.float32(rpss)
fid=open("./cca_rpss_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(fprobs20)
fid=open("./cca_20th_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(fprobs80)
fid=open("./cca_80th_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(fprobs10)
fid=open("./cca_10th_${wk}.bin",'wb');
fout.tofile(fid); fid.close();

fout=np.float32(fprobs90)
fid=open("./cca_90th_${wk}.bin",'wb');
fout.tofile(fid); fid.close();


eofPY

/cpc/home/ebekele/.conda/envs/xcast_env/bin/python gen_cca.py

cat>cca_ann_${wk}.ctl<<eofCTL
dset ^cca_ann_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
prob 0 99 Probability
endvars
eofCTL

cat>cca_nnn_${wk}.ctl<<eofCTL
dset ^cca_nnn_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
prob 0 99 Probability
endvars
eofCTL

cat>cca_bnn_${wk}.ctl<<eofCTL
dset ^cca_bnn_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
prob 0 99 Probability
endvars
eofCTL

cat>cca_groc_${wk}.ctl<<eofCTL
dset ^cca_groc_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
rr 0 99 groc
endvars
eofCTL

cat>cca_rpss_${wk}.ctl<<eofCTL
dset ^cca_rpss_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
rr 0 99 groc
endvars
eofCTL

for th in 10 20 80 90; do
cat>cca_${th}th_${wk}.ctl<<eofCTL
dset ^cca_${th}th_${wk}.bin
undef -999000000.0000000.000000
xdef 105 linear 155 0.25
ydef 93 linear -20 0.25
zdef 1 linear 1 1
tdef 1 linear 01oct2021 1mon
vars 1
prob 0 99 groc
endvars
eofCTL
done


cat>gen_tercile.gs<<eofGS
'reinit'
'open /cpc/home/ebekele/gen_mask_for_grads/pacific.ctl'
'set lat -20 3'
'set lon 155 181'
'define mm = mask'
'close 1'
'open cca_20th_${wk}.ctl'
'open cca_80th_${wk}.ctl'
'open cca_10th_${wk}.ctl'
'open cca_90th_${wk}.ctl'

'set lat -20 3'
'set lon 155 181'
'define pp20=maskout(lterp(prob.1,mm),mm)*100'
'define pp80=maskout(lterp(prob.2,mm),mm)*100'
'define pp10=maskout(lterp(prob.3,mm),mm)*100'
'define pp90=maskout(lterp(prob.4,mm),mm)*100'
'close 4'
'close 3'
'close 2'
'close 1'

'open cca_ann_${wk}.ctl'
'open cca_nnn_${wk}.ctl'
'open cca_bnn_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define pan = maskout(lterp(prob.1,mm),mm)*100'
'define pnn = maskout(lterp(prob.2,mm),mm)*100'
'define pbn = maskout(lterp(prob.3,mm),mm)*100'
'define an = maskout(maskout(pan,pan-pnn),pan-pbn)'
'define nn = maskout(maskout(pnn,pnn-pan),pnn-pbn)'
'define bn = maskout(maskout(pbn,pbn-pan),pbn-pbn)'
'define an80 = maskout(maskout(pp80,pp80-pp20),(100-(pp80+pp20)))'
'define bn20 = maskout(maskout(pp20,pp20-pp80),(100-(pp80+pp20)))'
'define an90 = maskout(maskout(pp90,pp90-pp10),(100-(pp90+pp10)))'
'define bn10 = maskout(maskout(pp10,pp10-pp90),(100-(pp90+pp10)))'

'define avanchk = aave(an,global)'
'define avnnchk = aave(nn,global)'
'define avbnchk = aave(bn,global)'

'd avanchk'; res1 = sublin(result,1); undan = subwrd(res1,4)
'd avnnchk'; res2 = sublin(result,1); undnn = subwrd(res2,4)
'd avbnchk'; res3 = sublin(result,1); undbn = subwrd(res3,4)

'define av20chk = aave(pp20,global)'
'define av80chk = aave(pp80,global)'
'define av10chk = aave(pp10,global)'
'define av90chk = aave(pp90,global)'

'd av20chk'; res1 = sublin(result,1); und20 = subwrd(res1,4)
'd av80chk'; res2 = sublin(result,1); und80 = subwrd(res2,4)
'd av10chk'; res1 = sublin(result,1); und10 = subwrd(res1,4)
'd av90chk'; res2 = sublin(result,1); und90 = subwrd(res2,4)


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

'set mpdraw off'
'set display color white'
'c'
'set xlint 5'
'set ylint 5'
'set grads off'
if(und80>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 30 29 28 27 26 25 24 23 22 21'

'd an80'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 6.35 8.85 0.25 0.45'
endif
if(und20>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 33 34 35 36 37 38 39 40 41 42'
'd bn20'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 1.85 4.35 0.25 0.45'
endif
'set line 1 1 6'
'draw shp /cpc/home/ebekele/packages/subseason/gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.12'
'draw string 5.5 7.9 GEFS, Week-${wk}, CCA, <20th / >80th, alid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cca20N80.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cca20N80.png gefs_week_${wk}_cca20N80.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cca20N80.png gefs_week_${wk}_cca20N80.png'
'c'

'set xlint 5'
'set ylint 5'
'set grads off'
if(und90>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 30 29 28 27 26 25 24 23 22 21'

'd an90'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 6.35 8.85 0.25 0.45'
endif
if(und10>0)
'set clevs 40 45 50 55 60 65 70 75 80'
'set ccols 33 34 35 36 37 38 39 40 41 42'
'd bn10'
'/cpc/home/ebekele/packages/subseason/gradssupp/xbar.gs -fwidth 0.10 -line on -edge triangle 1.85 4.35 0.25 0.45'
endif
'set line 1 1 6'
'draw shp /cpc/home/ebekele/packages/subseason/gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.12'
'draw string 5.5 7.9 GEFS, Week-${wk}, CCA, <10th / >90th, alid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cca10N90.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cca10N90.png gefs_week_${wk}_cca10N90.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cca10N90.png gefs_week_${wk}_cca10N90.png'
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
'set strsiz 0.14'
'draw string 5.5 7.9 GEFS, Week-${wk}, CCA, Valid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cca.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cca.png gefs_week_${wk}_cca.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cca.png gefs_week_${wk}_cca.png'
'c'


'quit'
eofGS

/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads -blc gen_tercile.gs

cat>gen_groc_rpss.gs<<eofGS
'reinit'
'open /cpc/home/ebekele/gen_mask_for_grads/pacific.ctl'
'set lat -20 3'
'set lon 155 181'
'define mm = mask'
'close 1'
'open cca_groc_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define pp = maskout(lterp(rr,mm),mm)'
'set gxout shaded'
'set mpdraw off'
'/cpc/home/ebekele/packages/subseason/gradssupp/define_colors'
'set display color white'
'c'
'set grads off'
'set xlint 5'
'set ylint 5'
'set clevs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9'
'set ccols 49 47 45 43 41 61 63 65 67 69'
'd pp'
'/cpc/home/ebekele/packages/subseason/gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp /cpc/home/ebekele/packages/subseason/gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.13'
'draw string 5.5 7.9 GEFS, Week-${wk}, CCA-GROC, Valid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cca_groc.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cca_groc.png gefs_week_${wk}_cca_groc.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cca_groc.png gefs_week_${wk}_cca_groc.png'
'c'
'close 1'
'open cca_rpss_${wk}.ctl'
'set lat -20 3'
'set lon 155 181'
'define pp = maskout(lterp(rr,mm),mm)'
'set gxout shaded'
'set mpdraw off'
'/cpc/home/ebekele/packages/subseason/gradssupp/define_colors'
'set display color white'
'c'
'set grads off'
'set xlint 5'
'set ylint 5'
'set clevs -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5'
'set ccols 49 48 47 46 44 42 62 64 66 67 68 69'
'd pp'
'/cpc/home/ebekele/packages/subseason/gradssupp/cbarmerc2.gs'
'set line 1 1 6'
'draw shp /cpc/home/ebekele/packages/subseason/gradssupp/WMO_basemap.shp'
'set string 1 c'
'set strsiz 0.13'
'draw string 5.5 7.9 GEFS, Week-${wk}, ELR-RPSS, Valid: ${iwk} - ${fwk}'
'printim gefs_week_${wk}_cca_rpss.png  x1500 y1500'
'!convert -trim gefs_week_${wk}_cca_rpss.png gefs_week_${wk}_cca_rpss.png'
'!convert -bordercolor white -border 10 gefs_week_${wk}_cca_rpss.png gefs_week_${wk}_cca_rpss.png'
'quit'

eofGS

/cpc/home/ebekele/grads2.1/grads-2.1.0/bin/grads -blc gen_groc_rpss.gs

done
