# Coral Dev Board with Pose Estimation

Pose estimation seems to work decently well with MoveNet Thunder. 
Has sensitivity to lighting and background, if you can see the full person, and see them clearly, estimation appears pretty reliable. 

Stock (Mendel install)[https://coral.ai/docs/dev-board/get-started/#flash-the-board] with stock python

OTG port connection: 
`ls /dev/cu.usbmodem*`    (mac)
shows devices…
`screen <device> 115200`
Press enter to see login, mendel/mendel

Crtrl + A, K, then enter to end session


`nmcli dev wifi connect catlin password XXXXX`

`python3.7 -m venv --system-site-packages pose_venv`
`source ~/pose_venv/bin/activate`
`pip install --upgrade pip`  (to version 24.1)

`sudo apt install python3-opencv`

This device has a ridiculous requirement, no power button but it "must" be powered off by software or it may corrupt OS. 
Adding a digital button helps, code example included. GPIO13 to Button to Resistor to 3V3 Pin 17. 
`@reboot /usr/bin/python3 /home/mendel/power_button.py`


Versions in base:
```
Python 3.7
certifi==2018.8.24
chardet==3.0.4
distro-info==0.21
edgetpuvision==7.0
httplib2==0.11.3
idna==2.6
netifaces==0.10.4
numpy==1.16.2
Pillow==5.4.1
pip==18.1
protobuf==3.6.1
pycairo==1.16.2
pycoral==2.0.0
pycurl==7.43.0.2
PyGObject==3.30.4
PyJWT==1.7.0
PyOpenGL==3.1.0
pyserial==3.4
PySimpleSOAP==1.16.2
```
