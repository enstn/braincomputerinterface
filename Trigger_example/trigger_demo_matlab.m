%% Open TCPIP port to sync trigger with BioWolfGUI.
ConnParams = tcpip('127.0.0.1', 30000, 'NetworkRole', 'server');
fopen(ConnParams);


%% Make sure trigger starts with Zero.
fwrite(ConnParams, 0, 'uint8', 'sync');

for inx=1: 20
    fwrite(ConnParams, inx, 'uint8', 'sync');
    pause(2);
    fwrite(ConnParams, 0, 'uint8', 'sync');
    pause(2)
    disp("Experiment...")
end



