for e in 're1k' 're500'; do 
for t in '1' '2' '3'; do
 
./launchDaint.sh smartEll_VRAC_${e}_TRIAL${t} 16 smartCyl_${e} settings/settings_VRACER_expensiveData.sh 15
./launchDaint.sh smartEll_uDPG_${e}_TRIAL${t} 16 smartCyl_${e} settings/settings_DPG_unb_expensiveData.sh 15
./launchDaint.sh smartEll_bDPG_${e}_TRIAL${t} 16 smartCyl_${e} settings/settings_DPG_expensiveData.sh 15
./launchDaint.sh smartEll_ACER_${e}_TRIAL${t} 16 smartCyl_${e} settings/settings_ACER_expensiveData.sh 15

done
done
