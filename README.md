# 1DL550_Assignment
Assignment of 1DL550 Low-level Parallel Programming at Uppsala University

## Prerequisite: Qt5
Qt5 is required to build and run the assignment. Tested with Qt5 on Ubuntu
20.04. Installed Qt5 with the following command:

```
# apt install qtbase5-dev qtbase5-dev-tools
```

If you cannot install Qt5, but you do have Qt4 then you will need to modify
the include and library arguments in `demo/Makefile`. Change the `qt5` strings
under `QTINCLUDES` and `LIBS` to your Qt version.

## Building
Once QT5 has beein installed, run the makefile.

```
$ make
```

This should build both binaries in `libpedsim` then in `demo`

If during the build the compiler complains of unknown path to the Qt headers
(include path), check the path in `QTINCLUDES` and make sure they exist on
your system. Your system may put the include files at a different location.
The makefile currently tries to locate the Qt header files using the following
command:
```
$ qmake -query QT_INSTALL_HEADERS
```

If this fails, try locate the header files using the following command:
```
$ find / -name QTHEADER.h` (where the QTHEADER.h the compiler is looking for)
```
Using the results fix the path in `QT_HEADERS` variable in the `demo/Makefile`


## Running
If the build is successful, run the simulator using the following command

```
$ demo/demo scenario.xml
```
