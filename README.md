# faceswitch

## To open virtual env:

Kivy venv to run Kivy program locally:
``` source /mnt/c/Users/willc/OneDrive/docs/uni/ENGG4812/Code/switch/kivy_venv/Scripts/activate ```

Buildozer venv to build the android app:
``` source /mnt/c/Users/willc/OneDrive/docs/uni/ENGG4812/Code/switch/buildozer/venv-buildozer/bin/activate ```


## JAVA

- Needs to be java-1.11.0-openjdk-amd64

``` export JAVA_HOME='/usr/lib/jvm/java-1.8.0-openjdk-amd64' ```
``` export PATH=$JAVA_HOME:$PATH ```



## Kivy and Buildozer:

1. https://kivy.org/doc/stable/gettingstarted/installation.html
2. https://kivy.org/doc/stable/guide/packaging-android.html
3. https://buildozer.readthedocs.io/en/latest/installation.html#targeting-android
4. https://buildozer.readthedocs.io/en/latest/quickstart.html

- To save the logcat output into a file named my_log.txt (the file will appear in your current directory):

``` buildozer -v android debug deploy run logcat > my_log.txt ```

- To see your running application’s print() messages and python’s error messages, use:

``` buildozer -v android debug deploy run logcat | grep python ```

- If, after changing the requirements in buildozer.spec you still see your app crashing, run the following command(s)

``` buildozer android clean ```


## Connect to device:

1. Device must allow files and be in USB debugging mode

2. On Windows:

``` E:/Users/willc/AppData/Local/Android/platform-tools/adb kill-server ```
``` E:/Users/willc/AppData/Local/Android/platform-tools/adb tcpip 5555 ```

- If devices isn't showing, use the 'kill-server' and 'start-server' commands

3. Then on WSL 2:

``` /home/will12193/.buildozer/android/platform/android-sdk/platform-tools/adb connect 192.168.0.17:5555 ```

- You can get your [ip device] from: Settings > About device > Status. It has a form like: 192.168.1.100