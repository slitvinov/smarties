# CAREFULL, the first fish has to be placed at xpos=L/2, the second is assumed to have fixed length and follows at xpos=3L/2
../launchSim.sh -bFreeSpace 0 -muteAll 0 -bpdx 32 -bpdy 16 -tdump 0.1 -nu 0.00001 -tend 0 -shapes 'stefanfish L=0.4 xpos=0.2 bFixed=1 pid=1,stefanfish L=0.2 xpos=0.6 bFixedy=1'
