QT       += core gui printsupport multimedia concurrent

# charts

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# You can make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    myDFT.cpp \
    qcustomplot.cpp

HEADERS += \
    MusicPlayer.h \
    mainwindow.h \
    myDFT.h \
    qcustomplot.h

FORMS += \
    mainwindow.ui

#INCLUDEPATH += D:/Qt/qt_pro/Image_Processing/opencv/out/include

#DEPENDPATH += D:/Qt/qt_pro/Image_Processing/opencv/out/include

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#win32:CONFIG(release, debug|release): LIBS += -LD:/Qt/qt_pro/Image_Processing/opencv/build/x64/vc15/lib/ -lopencv_world460
#else:win32:CONFIG(debug, debug|release): LIBS += -LD:/Qt/qt_pro/Image_Processing/opencv/build/x64/vc15/lib/ -lopencv_world460d
#else:unix: LIBS += -LD:/openCV/opencv/build/x64/vc15/lib/ -lopencv_world460

RESOURCES += \
    res.qrc


# debug
#win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../opencv/out/x64/mingw/lib/ -llibopencv_world460d.dll
#else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../opencv/out/x64/mingw/lib/ -llibopencv_world460d.dll
#else:unix: LIBS += -L$$PWD/../opencv/out/x64/mingw/lib/ -llibopencv_world460d.dll

#INCLUDEPATH += $$PWD/../opencv/releaseout/include
#DEPENDPATH += $$PWD/../opencv/releaseout/include

# release
win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../opencv/releaseout/x64/mingw/lib/ -llibopencv_world460.dll
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../opencv/releaseout/x64/mingw/lib/ -llibopencv_world460.dll
else:unix: LIBS += -L$$PWD/../opencv/releaseout/x64/mingw/lib/ -llibopencv_world460.dll

INCLUDEPATH += $$PWD/../opencv/releaseout/include
DEPENDPATH += $$PWD/../opencv/releaseout/include


#win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../opencv/out/x64/mingw/lib/liblibopencv_world460d.dll.a
#else:win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../opencv/out/x64/mingw/lib/liblibopencv_world460d.dlld.a
#else:win32:!win32-g++:CONFIG(release, debug|release): PRE_TARGETDEPS += $$PWD/../opencv/out/x64/mingw/lib/libopencv_world460d.dll.lib
#else:win32:!win32-g++:CONFIG(debug, debug|release): PRE_TARGETDEPS += $$PWD/../opencv/out/x64/mingw/lib/libopencv_world460d.dlld.lib
#else:unix: PRE_TARGETDEPS += $$PWD/../opencv/out/x64/mingw/lib/liblibopencv_world460d.dll.a


