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

    /cpc/home/kkowal/.conda/envs/map_env/bin/python /cpc/int_desk/pac_isl/stations/updated_images_for_leaflet.py
    cd /cpc/int_desk/pac_isl/stations/
    current_branch=$temp-branch
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
             