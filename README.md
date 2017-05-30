# FenceIndoor
<br>
..explained in three simple steps:
<table border="0" width="100%">
<tr><td>
<a href="Screenshots/step1.png"><img src="Screenshots/step1.png" width=400></a>
</td><td valign=top>
Step 1:<br>
Select the area you are in and capture all WiFi signals around device
</td></tr>
<tr><td>
<a href="Screenshots/step2.png"><img src="Screenshots/step2.png" width=400></a>
</td><td valign=top>
Step 2:<br>
Train the artificial neural network with the collected data
</td></tr>
<tr><td>
<a href="Screenshots/step3.png"><img src="Screenshots/step3.png" width=400></a>
</td><td valign=top>
Step 3:<br>
As you move the neural network will make a prediction of the area you are in, 
depending on the wifi signals caught around the device
</td></tr>
</table>
<br><br>
Sceenshots
<table border="0" width="100%">
<tr><td>
<a href="Screenshots/home.png"><img src="Screenshots/home.png" width=120></a>
</td><td>
<a href="Screenshots/areaList.png"><img src="Screenshots/areaList.png" width=120></a>
</td><td>
<a href="Screenshots/wifiScans.png"><img src="Screenshots/wifiScans.png" width=120></a>
</td><td>
<a href="Screenshots/predict.png"><img src="Screenshots/predict.png" width=120></a>
</td></tr>
</table>
<br><br>
Installation:

From the project directory..


- Generate apk:

> cd FendeIndoorApp

> ./gradlew assemble

made apk to the path:

> ./FendeIndoorApp/app/build/outputs/apk/app-release-unsigned.apk


- Start server:

> cd FenceIndoorServer
> python fenceIndoor.py

