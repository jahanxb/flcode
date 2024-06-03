@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET codePath=C:\Users\mkhan40\Documents\vscodeprojects\flcode

FOR /L %%i IN (1,1,10) DO (
    SET /A ipSuffix=%%i + 1
    SET ip=172.18.0.!ipSuffix!
    docker run -d --name ubuntu_container_%%i --net mynetwork --ip !ip! -v %codePath%:/opt myubuntuimage
)

ENDLOCAL
