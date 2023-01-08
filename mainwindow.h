#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QString>
#include <QImage>
#include <QFileDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QFileInfo>
#include <QPixmap>
#include <QDebug>
#include <QMessageBox>
#include <QAction>
#include <QLinearGradient>
#include <QMatrix>
#include <QTransform>
#include <QLabel>
#include <QGraphicsOpacityEffect>
#include <QDateTime>
#include <QSlider>
#include <QtMath>
#include <time.h>
#include <complex>

#include <QUrl>
#include <QMediaPlayer>
#include <QThread>
#include <QFuture>
#include <QtConcurrent>
#include <QListIterator>
//#include <QMediaPlaylist>

#include <iostream>
#include <QVector>
#include <QQueue>
#include <QMap>
#include <QList>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h> // CV_BGR2RGB 引用报错

#include "qcustomplot.h"
#include "MusicPlayer.h"
#include "myDFT.h"

//#include "../FFTW/fftw-3.3.5-dll32/fftw3.h"
//#pragma comment(lib, "../FFTW/fftw-3.3.5-dll32/libfftw3-3.dll")

//QT_CHARTS_USE_NAMESPACE

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

using namespace cv;

//typedef std::complex<double> Complex;

struct ImageNode {
    QImage img;
    int label_idx;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

signals:
    void signal_combox2(int);

public slots:
    void on_OpenImg();
    void on_MusicPlay();
    void on_showOringinImg();
    void on_showGrayImg();
    void on_saveImg();
    void on_ComBox1(int index);
    void on_ComBox2(int index);
    void on_ComBox3(int index);
    void on_ComBox4(int index);

    // 右侧dockwidgt的槽函数
    void on_bright_valueChanged(int value);
    void on_binary_valueChanged(int value);
    void on_R_valueChanged(int value);
    void on_G_valueChanged(int value);
    void on_B_valueChanged(int value);

    // 底部stateBar时间动态实时显示
    void on_updateDateTime();

public:                 // public fun
    // 功能型函数
    void Setting();
    void LabelIdxToSetPixmap(QImage img, int label_idx);
    QImage Mat2QImage(const cv::Mat& mat);
    cv::Mat QImage2Mat(const QImage& image);
//    void fft(Complex* data, int n);
//    void fft(QImage& image);

    // 功能一
    void showHistogram(QImage image);

    void func1_imadjust(double alpha = -1, double beta = 20);   // 线性变换
    void func1_binary(int threshold = 131);                     // 二值化
    void func1_gamma(double alpha = 1, double gamma = 0.5);     // 伽马变换
    void func1_grayavg();                                       // 灰度均衡化
    void func1_log(double alpha = 1);                           // 对数处理
    void func1_power(double power = 1);                         // 幂数变换
    void func1_reverse();                                       // 图像反色

    // 功能二
    void func2_mean();     // 均值滤波
    void func2_median();   // 中值滤波
    void func2_gaussian(); // 高斯滤波
    void func2_bilateral();// 双边滤波
    void func2_laplacian();// 拉普拉斯
    void func2_max();      // 最大值滤波
    void func2_sobel();    // Sobel梯度算子

    void func2_difference(QImage &image);
    void func2_hysteresisThreshold(QImage& image, int lowThreshold, int highThreshold);

    // 功能三
    void func3_pepper();    // 添加椒盐噪声
    void func3_lowpass(float sigma = 30);   // 理想低通滤波
    void func3_highpass(float sigma = 80);  // 理想高通滤波
    void func3_gaussian_lowpass(float sigma = 30); // 高斯低通滤波
    void func3_gaussian_highpass(float sigma = 80); // 高斯高通滤波

    // 功能四
    void func4_pseudocolor();
    void func4_edge();

private:                // private var
    QString originPath;
    QImage originImg;
    QImage grayImg;
    int lastBright, lastR, lastG, lastB;
    QMap<int, QList<ImageNode>> tab;

    // 功能一
    QImage imadjustImg, logImg, binaryImg, gammaImg, grayavgImg, powerImg, reverseImg;

    QImage sharpImg;

    // 功能二
    QImage centerImg, leftImg, rightImg, mainImg; // RGB滑动条，将centerImg,作为原图像的备份，mainImg,实时改变。
    int cursor, kernel; // 默认进入是1开始 state（1 ~ 7）
    int last_cursor;    // 用于对widget的红边框清空
    QString mainImg_name[8] = {"", "均值滤波", "中值滤波", "高斯滤波", "双边滤波", "拉普拉斯算子", "最大值滤波", "Sobel算子"};

    // 功能三
    QImage leftImg3;

    // 功能四
    QImage mainImg4;

    QCustomPlot *plot;

    // 底部stateBar定时器
    QTimer *timer;

public:                 // public var
    // Music
    bool is_play;
    MusicPlayer* musicPlayer;
//    QMediaPlaylist musicPlayList;
};
#endif // MAINWINDOW_H
