#!/bin/bash

#Script: station_image_prep.sh
#Description: prepare image files for interactive map over Pacific Islands
#Author: Katie Kowal
#Date: 7/11/24

#set up the environment
#export #PATH="/cpc/home/kkowal/.conda/envs/map_env/bin/python"

#Log file (to capture output)
LOGFILE="/cpc/int_desk/pac_isl/stations/images/station_image_prep.log"
figure_dir=/cpc/int_desk/pac_isl/stations/images
#function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> "$LOGFILE"
}

#main function
main(){
    log "starting cron task"
    #runs the download script for OISSTv2 to get latest day available
    /cpc/home/kkowal/.conda/envs/xcast_env/bin/python /cpc/int_desk/data/oisstv2/download_oisstv2.py
    
    #pulls and crops the updated png files for the leaflet html file - legends, titles, etc...
    /cpc/home/kkowal/.conda/envs/map_env/bin/python /cpc/int_desk/pac_isl/stations/updated_images_for_leaflet.py
    
    #prepares the oisst sst data, calculates anomalies for tile generation
    /cpc/home/kkowal/.conda/envs/map_env/bin/python /cpc/int_desk/pac_isl/stations/prep_sst_data.py
    
    #prepares the cmorph data, calculates anomalies for tiles generation
    /cpc/home/kkowal/.conda/envs/map_env/bin/python /cpc/int_desk/pac_isl/stations/prep_cmorph_data.py
    
    #prepares the gefs data, calcualtes anomaleis for tile generation
    /cpc/home/kkowal/.conda/envs/map_env/bin/python /cpc/int_desk/pac_isl/stations/prep_gefs.py
    
    cd /cpc/int_desk/pac_isl/stations/
    #tile generation codes - take a tif and convert to tiles in mercator projection
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/sst_mercator7.tif /cpc/int_desk/pac_isl/stations/images/sst_anom7_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/sst_mercator7diff.tif /cpc/int_desk/pac_isl/stations/images/sst_anom7diff_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph90anom.tif /cpc/int_desk/pac_isl/stations/images/cmorph90anom_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph90percent.tif /cpc/int_desk/pac_isl/stations/images/cmorph90percent_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph30anom.tif /cpc/int_desk/pac_isl/stations/images/cmorph30anom_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph30percent.tif /cpc/int_desk/pac_isl/stations/images/cmorph30percent_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph7anom.tif /cpc/int_desk/pac_isl/stations/images/cmorph7anom_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/cmorph7total.tif /cpc/int_desk/pac_isl/stations/images/cmorph7total_tiles
    
    current_branch=temp-branch
    log "Current branch: $current_branch"
    git add .
    git commit -m "automated commit by cron job on $(date)"
    git push origin "$current_branch"
    
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1pcons.tif /cpc/int_desk/pac_isl/stations/images/gefswk1pcons_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2pcons.tif /cpc/int_desk/pac_isl/stations/images/gefswk2pcons_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1ptotal.tif /cpc/int_desk/pac_isl/stations/images/gefswk1ptotalraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1panom.tif /cpc/int_desk/pac_isl/stations/images/gefswk1panomraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2ptotal.tif /cpc/int_desk/pac_isl/stations/images/gefswk2ptotalraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2panom.tif /cpc/int_desk/pac_isl/stations/images/gefswk2panomraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1ttotal.tif /cpc/int_desk/pac_isl/stations/images/gefswk1ttotalraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1tanom.tif /cpc/int_desk/pac_isl/stations/images/gefswk1tanomraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2ttotal.tif /cpc/int_desk/pac_isl/stations/images/gefswk2ttotalraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2tanom.tif /cpc/int_desk/pac_isl/stations/images/gefswk2tanomraw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1poe50.tif /cpc/int_desk/pac_isl/stations/images/gefswk1poe50raw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk1poe100.tif /cpc/int_desk/pac_isl/stations/images/gefswk1poe100raw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2poe50.tif /cpc/int_desk/pac_isl/stations/images/gefswk2poe50raw_tiles
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/gefswk2poe100.tif /cpc/int_desk/pac_isl/stations/images/gefswk2poe100raw_tiles
    
    
    #li xu tiles -- uncomment when color becomes available
    # gdal_translate -ot Byte -scale /cpc/int_desk/LiXu/geotiff/SPI3.tif /cpc/int_desk/pac_isl/stations/images/SPI3mo_8bit.tif
    # gdalwarp -s_srs EPSG:4326 -t_srs EPSG:3857 /cpc/int_desk/pac_isl/stations/images/SPI3mo_8bit.tif /cpc/int_desk/pac_isl/stations/images/SPI3mo_mc.tif
    
    ## trial one out
    gdal2tiles.py -p mercator -z 0-5 /cpc/int_desk/pac_isl/stations/images/SPI3mo_mc.tif /cpc/int_desk/pac_isl/stations/images/SPI3mo_tiles

    current_branch=temp-branch
    log "Current branch: $current_branch"
    git add .
    git commit -m "automated commit by cron job on $(date)"
    git push origin "$current_branch"

    # Check if the task was successful
    if [[ $? -eq 0 ]]; then
        log "Task completed successfully."
    else
        log "Task failed with errors."
    fi

    log "Cron task finished."
}

main