#for e in 're1k' ; do
for e in 're500'; do
#for t in '1' '2' '3'; do
for t in '14'; do

#./launch.sh smartCyl_new_002 smartCyl_new settings/settings_VRACER_expensiveData.sh 15 1 16 18
#./launch.sh smartCyl_new_002 smartCyl_new settings/settings_VRACER_expensiveData.sh 15 1 16 18
#./launch.sh smartCyl_new_002 smartCyl_new settings/settings_VRACER_expensiveData.sh 15 1 16 18
#./launch.sh smartCyl_new_002 smartCyl_new settings/settings_VRACER_expensiveData.sh 15 1 16 18

#./launch.sh smartEllEfficControl_VRAC_${e}_TRIAL${t} smartCyl_${e} settings/settings_VRACER_expensiveData.sh  23 1 24 18
./launch.sh smartEllEfficControl_VRAC_${e}_TRIAL${t} smartCyl settings/settings_VRACER_expensiveData.sh  23 1 24 18
#./launch.sh smartEllEfficControl_VRAC_${e}_TRIAL${t} smartCyl_${e} settings/settings_VRACER_expensiveData.sh  1 1 2 18
#./launch.sh smartEllEffic0_VRAC_${e}_TRIAL${t} smartCyl_${e} settings/settings_VRACER_expensiveData.sh  11 1 12 18
#./launch.sh smartEllEffic0_uDPG_${e}_TRIAL${t} smartCyl_${e} settings/settings_DPG_unb_expensiveData.sh 11 1 12 18
#./launch.sh smartEllEffic0_bDPG_${e}_TRIAL${t} smartCyl_${e} settings/settings_DPG_expensiveData.sh     11 1 12 18
#./launch.sh smartEllEffic0_ACER_${e}_TRIAL${t} smartCyl_${e} settings/settings_ACER_expensiveData.sh    11 1 12 18

done
done
