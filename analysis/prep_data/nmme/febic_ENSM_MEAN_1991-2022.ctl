dset /cpc/int_desk/NMME/hindcast/raw_sst_precip_tmp2m/precip_monthly/febic_ENSM_MEAN_1991-2022.dat
undef 9.999E+20
title prate.bin
options little_endian
xdef 360 linear 0 1.0
ydef 181 linear -90.0 1.0
tdef 32 linear 15feb1991 1yr
zdef 9 linear 1 1
vars 1
fcst 9,1,0   0,1,7,0 ** sst DegC
ENDVARS
