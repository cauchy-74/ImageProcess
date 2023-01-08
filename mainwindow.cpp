#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , lastBright(0)
    , lastR(0)
    , lastG(0)
    , lastB(0)
    , cursor(1)
    , kernel(3)
    , last_cursor(0)
    , timer(new QTimer(this))
    , is_play(false)
    , musicPlayer(new MusicPlayer(this))
{
    ui->setupUi(this);
    this->Setting(); srand(unsigned(time(NULL)));

    // 左侧dockwidget
    connect(ui->btn_choose_file, &QPushButton::clicked, this, [&](){ QtConcurrent::run([&](){ this->on_OpenImg(); }); });
    connect(ui->btn_gray_image, &QPushButton::clicked, this, [&](){ this->on_showGrayImg(); });
    connect(ui->btn_origin_image, &QPushButton::clicked, this, [&](){ this->on_showOringinImg(); });
    /*connect(ui->btn_origin_image, &QPushButton::clicked, this, [&](){ ui->label_origin_image->setPixmap(QPixmap::fromImage(originImg));}); */
    connect(ui->btn_music, &QPushButton::clicked, this, [&](){ this->on_MusicPlay(); });
    connect(ui->comboBox, SIGNAL(currentIndexChanged(int)), this, SLOT(on_ComBox1(int)));
        // 第二个功能
    connect(this, SIGNAL(signal_combox2(int)), this, SLOT(on_ComBox2(int)));
    connect(ui->comboBox2, SIGNAL(currentIndexChanged(int)), this, SLOT(on_ComBox2(int)));
        // 左右按钮
    connect(ui->btn_left, &QPushButton::clicked, [&](){ emit signal_combox2(cursor - 1); });
    connect(ui->btn_right, &QPushButton::clicked, [&](){ emit signal_combox2(cursor + 1); });

        // 第三个功能
    connect(ui->comboBox3, SIGNAL(currentIndexChanged(int)), this, SLOT(on_ComBox3(int)));

        // 第四个功能
    connect(ui->comboBox4, SIGNAL(currentIndexChanged(int)), this, SLOT(on_ComBox4(int)));
    connect(ui->btn_save_image, &QPushButton::clicked, this, &MainWindow::on_saveImg);

    // 右侧dockwidget
    connect(ui->slider_bright, SIGNAL(valueChanged(int)), this, SLOT(on_bright_valueChanged(int)));
    connect(ui->slider_binary, SIGNAL(valueChanged(int)), this, SLOT(on_binary_valueChanged(int)));
    connect(ui->slider_R, SIGNAL(valueChanged(int)), this, SLOT(on_R_valueChanged(int)));
    connect(ui->slider_G, SIGNAL(valueChanged(int)), this, SLOT(on_G_valueChanged(int)));
    connect(ui->slider_B, SIGNAL(valueChanged(int)), this, SLOT(on_B_valueChanged(int)));

    // 底部stateBar
    connect(timer, &QTimer::timeout, this, [&](){ QtConcurrent::run([&](){ this->on_updateDateTime(); }); }); timer->start(1000);

}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::showHistogram(QImage image) {
    qDebug() << Q_FUNC_INFO << "\n";
    if (image.isNull()) {
        return ;
    }
    if (!image.allGray()) {
        image = image.convertToFormat(QImage::Format_Grayscale8);
    }
    plot = new QCustomPlot(ui->label_gray_histogram);
    plot->clearGraphs();
    plot->addGraph();
    plot->graph(0)->setPen(QPen(Qt::blue));

    // 创建灰度直方图数组
    int histogram[256] = {0};

    for (int j = 0; j < image.height(); j++) {
        for (int i = 0; i < image.width(); i++) {
            // 统计灰度值的个数
            histogram[qGray(QRgb(image.pixel(i, j)))]++;
        }
    }
    // 添加柱状图
    QCPBars *bars = new QCPBars(plot->xAxis, plot->yAxis);

    // 设置柱状图的数据
    QVector<double> x(256), y(256);
    for (int i = 0; i < 256; i++)
    {
        x[i] = i;
        y[i] = histogram[i];
    }
    bars->setData(x, y);

    // 设置横坐标的标签
    QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
    for (int i = 0; i < 256; i += 10)
    {
        textTicker->addTick(i, QString::number(i));
    }
    textTicker->addTick(255, QString::number(255));
    plot->xAxis->setTicker(textTicker);

    // 设置坐标轴的范围
    plot->xAxis->setRange(0, 255);
    plot->yAxis->setRange(0, *std::max_element(histogram, histogram + 256));

    plot->resize(ui->label_gray_histogram->width(), ui->label_gray_histogram->height());
    plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom); // 设置可交互图像

    plot->replot();
    plot->show();
}

void MainWindow::func1_imadjust(double alpha, double beta) {
    imadjustImg = grayImg.copy(); //.convertToFormat(QImage::Format_ARGB32_Premultiplied);
    // Qt::IgnoreAspectRatio 指定忽略图像的宽高比;在缩放图像时，可能会出现图像变形的情况
    imadjustImg.scaled(ui->label_imadjust_image->size(), Qt::IgnoreAspectRatio);

    QColor oldColor;
    for (int j = 0;  j < imadjustImg.height(); j++) {
        for (int i = 0; i < imadjustImg.width(); i++) {
            oldColor = QColor(imadjustImg.pixel(i, j));
            int res = (int)(alpha * (double)(oldColor.red()) + beta);
            imadjustImg.setPixel(i, j, qRgb(res, res, res));
        }
    }

    ui->label_imadjust_image->setPixmap(QPixmap::fromImage(imadjustImg));
}

/**
 * @brief MainWindow::func1_log
 * @param alpha
 */
/*
 *          O = alpha * log(1 + I)
 *   对数变换是一种图像增强方法，它通过将图像的像素值进行对数变换来增强图像的对比度。
 */
void MainWindow::func1_log(double alpha) {
    logImg = grayImg.copy();
    logImg.scaled(ui->label_log_image->size(), Qt::IgnoreAspectRatio);

    for (int j = 0;  j < logImg.height(); j++) {
        for (int i = 0; i < logImg.width(); i++) {
            int gray = round(alpha * log(1 + qGray(logImg.pixel(i, j))) / log(256) * 256);
            logImg.setPixel(i, j, qRgb(gray, gray, gray));
        }
    }
    ui->label_log_image->setPixmap(QPixmap::fromImage(logImg));
}

/**
 * @brief MainWindow::func1_power
 * @param power
 */
/*
    幂数变换： 非线性变换，改变 对比度、亮度
        power值越大，图像的对比度越高，亮度越低；
        power值越小，图像的对比度越低，亮度越高
        power值在0.5~1.5之间的效果比较好
*/
void MainWindow::func1_power(double power) {
    powerImg = grayImg.copy();
    powerImg.scaled(ui->label_power_image->size(), Qt::IgnoreAspectRatio);
    // 计算图像的最大和最小像素值
    int mn = 255;
    int mx = 0;
    for (int y = 0; y < powerImg.height(); y++) {
        for (int x = 0; x < powerImg.width(); x++) {
            int gray = qGray(powerImg.pixel(x, y));
            mn = qMin(mn, gray);
            mx = qMax(mx, gray);
        }
    }
    // alpha和beta的作用就是将图像的像素值映射到0~255这个范围内。
    // 这样做的好处是，变换后的图像看起来更好，能够更加清晰地展示出图像的细节
    double alpha = 255.0 / (mx - mn);   // 缩放系数
    double beta = -alpha * mn;          // 偏移量
    /* alpha * (mx - mn); */
    for (int y = 0; y < powerImg.height(); y++) {
        for (int x = 0; x < powerImg.width(); x++) {
            int gray = qPow(alpha * qGray(powerImg.pixel(x, y)) + beta, power);
            powerImg.setPixel(x, y, qRgb(gray, gray, gray));
        }
    }
    ui->label_power_image->setPixmap(QPixmap::fromImage(powerImg));
}

/**
 * @brief MainWindow::func1_reverse
 */
/*
    图像反色
*/
void MainWindow::func1_reverse() {
    reverseImg = grayImg.copy();
    reverseImg.scaled(ui->label_reverse_image->size(), Qt::IgnoreAspectRatio);

    // 将图像的每个像素的颜色取反
    for (int y = 0; y < reverseImg.height(); y++) {
        for (int x = 0; x < reverseImg.width(); x++) {
            int gray = qGray(reverseImg.pixel(x, y));
            reverseImg.setPixel(x, y, qRgb(255 - gray, 255 - gray, 255 - gray));
        }
    }
    ui->label_reverse_image->setPixmap(QPixmap::fromImage(reverseImg));
}

/**
 * @brief MainWindow::func1_binary
 */
void MainWindow::func1_binary(int threshold) {
    binaryImg = grayImg.copy();
    binaryImg.scaled(ui->label_binary_image->size(), Qt::IgnoreAspectRatio);

    for (int j = 0;  j < binaryImg.height(); j++) {
        for (int i = 0; i < binaryImg.width(); i++) {
            int gray = qGray(binaryImg.pixel(i, j));
            // 如果灰度值大于阈值，则将像素设为白色，否则设为黑色
            if (gray > threshold) {
                binaryImg.setPixel(i, j, qRgb(255, 255, 255));
            } else {
                binaryImg.setPixel(i, j, qRgb(0, 0, 0));
            }
        }
    }
    ui->label_binary_image->setPixmap(QPixmap::fromImage(binaryImg));
}

/**
 * @brief MainWindow::func1_gamma
 * @param alpha
 * @param gamma
 */
/*          O = K * (I)^γ
 *  伽马变换 ：线性变换，一种图像增强方法，对比度和亮度。
 *
 * K称为灰度缩放系数，用于整体拉伸图像灰度，通常取值为1
 *  当gamma >1时，伽马变换将拉低图像灰度值，图像视觉上变暗；
 *  当gamm<1.0时，伽马变换将提高图像的灰度值，图像视觉上变亮
 */
void MainWindow::func1_gamma(double alpha, double gamma) {
    gammaImg = grayImg.copy();
    gammaImg.scaled(ui->label_gamma_image->size(), Qt::IgnoreAspectRatio);

    for (int j = 0;  j < gammaImg.height(); j++) {
        for (int i = 0; i < gammaImg.width(); i++) {
            int gray = round(alpha * pow(qGray(gammaImg.pixel(i, j)) / 255.0, gamma) * 255);
            gammaImg.setPixel(i, j, qRgb(gray, gray, gray));
        }
    }
    ui->label_gamma_image->setPixmap(QPixmap::fromImage(gammaImg));
}

/**
 * @brief MainWindow::func1_grayavg
 */

/*          灰度均衡化
 *
   调用 QImage 的 convertToFormat() 方法，将图像转换为灰度图像。
   遍历图像中的每个像素，并统计每个灰度值出现的次数，存储在灰度直方图数组中。
   遍历图像中的每个像素，计算新的灰度值。
   调用 QImage 的 setPixel() 方法，更新图像中的每个像素。
 *
 */
void MainWindow::func1_grayavg() {
    grayavgImg = grayImg.copy();// .convertToFormat(QImage::Format_Grayscale8);
    grayavgImg.scaled(ui->label_grayavg_image->size(), Qt::IgnoreAspectRatio);

    int histogram[256] = {0};

    for (int j = 0; j < grayavgImg.height(); j++) {
        for (int i = 0; i < grayavgImg.width(); i++) {
            histogram[qGray(QRgb(grayavgImg.pixel(i, j)))]++;
        }
    }

    // 计算累计概率直方图
    int cum[256] = {0}; cum[0] = histogram[0];
    int total = grayavgImg.width() * grayavgImg.height();
    for (int i = 1; i < 256; i++)
    {
        cum[i] = cum[i - 1] + histogram[i];
    }

    // 遍历图像中的每个像素
    for (int y = 0; y < grayavgImg.height(); y++)
    {
        for (int x = 0; x < grayavgImg.width(); x++)
        {
            // 获取该像素的灰度值
            QRgb pixel = grayavgImg.pixel(x, y);
            int gray = qGray(pixel);

            // 计算新的灰度值
            int newGray = (int)(255.0 * cum[gray] / total + 0.5);

            // 更新图像中的每个像素
            grayavgImg.setPixel(x, y, qRgb(newGray, newGray, newGray));
        }
    }

    ui->label_grayavg_image->setPixmap(QPixmap::fromImage(grayavgImg));
}

/**
 * @brief MainWindow::func2_mean
 * @param kernel
 */
void MainWindow::func2_mean() {
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage tmpImg = mainImg.copy();
    int kernel2 = kernel * kernel;
    QRgb tmp;
    for (int y = kernel / 2; y < mainImg.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < mainImg.width() - kernel / 2; x++) {
            int r = 0, g = 0, b = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    tmp = tmpImg.pixel(x + i, y + j);
                    r += qRed(tmp);
                    g += qGreen(tmp);
                    b += qBlue(tmp);
                }
            }
            r = qBound(0.0, 1.0 * r / kernel2, 255.0);
            g = qBound(0.0, 1.0 * g / kernel2, 255.0);
            b = qBound(0.0, 1.0 * b / kernel2, 255.0);
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
//    ui->label_main_image->setPixmap(QPixmap::fromImage(mainImg));
}

void MainWindow::func2_median() {
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage tmpImg = mainImg.copy();
    int kernel2 = kernel * kernel;
    QVector<QRgb> v(kernel2);
    QVector<int> rgb(kernel2);
    for (int y = kernel / 2; y < mainImg.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < mainImg.width() - kernel / 2; x++) {
            int idx = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    v[idx++] = tmpImg.pixel(x + i, y + i);
                }
            }
            int r, g, b;
            for (int i = 0; i < 3; i++) {
                idx = 0;
                for (auto tmp: v) {
                    if (i == 0) rgb[idx++] = qRed(tmp);
                    if (i == 1) rgb[idx++] = qGreen(tmp);
                    if (i == 2) rgb[idx++] = qBlue(tmp);
                }
                std::sort(rgb.begin(), rgb.end());
                if (i == 0) r = rgb[kernel2 / 2];
                if (i == 1) g = rgb[kernel2 / 2];
                if (i == 2) b = rgb[kernel2 / 2];
            }
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
//    ui->label_main_image->setPixmap(QPixmap::fromImage(mainImg));
}

/////////////////////////////////////////////////////////////// 功能二 ////////////////////////////////////////////////////////////////

// 通过计算高斯矩阵来实现高斯滤波，然后对每个像素进行滤波计算，最后将滤波后的像素值设置到新的图像中
void MainWindow::func2_gaussian() {
    // 创建一个新的QImage图像，用于存储滤波后的结果
//        QImage result = QImage(mainImg.width(), mainImg.height(), mainImg.format());
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage result = mainImg.copy();
    // 计算滤波器的高斯矩阵
    double gaussKernel[9] = {
        0.075114, 0.123841, 0.075114,
        0.123841, 0.204180, 0.123841,
        0.075114, 0.123841, 0.075114 };
    unsigned imageKernel[3][9]; for (int i = 0; i < 3; i++) for (int j = 0; j < 9; j++) imageKernel[i][j] = 0;

    // 对每个像素进行滤波
    for (int y = kernel / 2; y < result.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < result.width() - kernel / 2; x++) {
            int cnt = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    QRgb tmp = result.pixel(x + i, y + j);
                    imageKernel[0][cnt] = qRed(tmp);
                    imageKernel[1][cnt] = qGreen(tmp);
                    imageKernel[2][cnt] = qBlue(tmp);
                    cnt++;
                }
            }
            double r = 0, g = 0, b = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 9; j++) {
                    if (i == 0) r += gaussKernel[j] * imageKernel[i][j];
                    if (i == 1) g += gaussKernel[j] * imageKernel[i][j];
                    if (i == 2) b += gaussKernel[j] * imageKernel[i][j];
                }
            }
            r = (int)qBound(0.0, r, 255.0);
            g = (int)qBound(0.0, g, 255.0);
            b = (int)qBound(0.0, b, 255.0);
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
}

// (src.type() == CV_8UC1 || src.type() == CV_8UC3) && src.data != dst.data) (之前的报错)
/**
 *  src: 源图像Mat对象，需要为8位或者浮点型单通道、三通道的图像
 *  dst: 目标图像Mat对象，不能直接用src来存储处理后的图像
 *  d: 表示在过滤过程中每个像素邻域的直径。如果这个值我们设其为非正数，那么OpenCV会从第五个参数sigmaSpace来计算出它来，在使用过程中我发现有点像模糊力度的意思。
 *  sigmaColor: 颜色空间滤波器的sigma值。
 *      (这个参数的值越大，就表明该像素邻域内有更宽广的颜色会被混合到一起，产生较大的半相等颜色区域。在使用过程中我发现有点像模糊范围的意思，范围越大看着越模糊)
 *  sigmaSpace: 坐标空间中滤波器的sigma值，坐标空间的标注方差
 *      (他的数值越大，意味着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。使用过程中我发现这个值越大，图像的过渡效果越好。
 *  borderType: 默认即可（BORDER_DEFAULT）
 */
void MainWindow::func2_bilateral() {
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_RGB888);
    cv::Mat src = QImage2Mat(mainImg);
    cv::Mat dst;
    cv::bilateralFilter(src, dst, 30, 35, 10);
    mainImg = Mat2QImage(dst);
}

void MainWindow::func2_laplacian() { //算子的系数之和需要为零
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage result = mainImg.copy();
    // 拉普拉斯锐化处理
    int LaplacianKernel[3][3] = {
        {-1, -1, -1},
        {-1, 8, -1},
        {-1, -1, -1}
    };
    QRgb tmp;
    for (int y = kernel / 2; y < result.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < result.width() - kernel / 2; x++) {
            int r = 0, g = 0, b = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    tmp = result.pixel(x + i, y + j);
                    r += qRed(tmp) * LaplacianKernel[kernel / 2 + i][kernel / 2 + j];
                    g += qGreen(tmp) * LaplacianKernel[kernel / 2 + i][kernel / 2 + j];
                    b += qBlue(tmp) * LaplacianKernel[kernel / 2 + i][kernel / 2 + j];
                }
            }
            r = qBound(0, r / kernel + qRed(tmp), 255);
            g = qBound(0, g / kernel + qGreen(tmp), 255);
            b = qBound(0, b / kernel + qBlue(tmp), 255);
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
}

void MainWindow::func2_max() {
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage result = mainImg.copy();
    QRgb tmp;
    for (int y = kernel / 2; y < result.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < result.width() - kernel / 2; x++) {
            int r = 0, g = 0, b = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    tmp = result.pixel(x + i, y + j);
                    r = std::max(r, qRed(tmp));
                    g = std::max(g, qGreen(tmp));
                    b = std::max(b, qBlue(tmp));
                }
            }
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
}

void MainWindow::func2_sobel() {
    int page = ui->tabWidget->currentIndex();
    mainImg = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    QImage result = mainImg.copy();
    // SobelX
    int KernelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
//    int KernelY[3][3] = {
//        {-1, -2, -1},
//        {0, 0, 0},
//        {1, 2, 1}
//    };
    QRgb tmp;
    for (int y = kernel / 2; y < result.height() - kernel / 2; y++) {
        for (int x = kernel / 2; x < result.width() - kernel / 2; x++) {
            int r = 0, g = 0, b = 0;
            for (int i = -kernel / 2; i <= kernel / 2; i++) {
                for (int j = -kernel / 2; j <= kernel / 2; j++) {
                    tmp = result.pixel(x + i, y + j);
                    r += qRed(tmp) * KernelX[kernel / 2 + i][kernel / 2 + j];
                    g += qGreen(tmp) * KernelX[kernel / 2 + i][kernel / 2 + j];
                    b += qBlue(tmp) * KernelX[kernel / 2 + i][kernel / 2 + j];
                }
            }
            r = qBound(0, r, 255);
            g = qBound(0, g, 255);
            b = qBound(0, b , 255);
            mainImg.setPixel(x, y, qRgb(r, g, b));
        }
    }
}

/**
 *  添加椒盐噪声
 */
void MainWindow::func3_pepper() {
    int page = ui->tabWidget->currentIndex();
    leftImg3 = (tab[page].at(0).img.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied);
    int pepper_count = 5000; // 椒盐个数
    int pepper_size = 6; // 椒盐大小
    int noise ;
    int xPosition, yPosition;
    while (pepper_count) {
        xPosition = rand() % (leftImg3.width() - pepper_size + 1);
        yPosition = rand() % (leftImg3.height() - pepper_size + 1);
        noise = rand() % 2;
        if (noise) {
            for (int i = 0; i < pepper_size; i++) {
                for (int j = 0; j < pepper_size; j++) {
                    leftImg3.setPixel(xPosition, yPosition, qRgb(255, 255, 255));
                }
            }
        } else {
            for (int i = 0; i < pepper_size; i++) {
                for (int j = 0; j < pepper_size; j++) {
                    leftImg3.setPixel(xPosition, yPosition, qRgb(0, 0, 0));
                }
            }
        }
        pepper_count -= 1;
    }
}

cv::Mat image_make_border(cv::Mat &src)
{
    int w = getOptimalDFTSize(src.cols);
    int h = getOptimalDFTSize(src.rows); //获取最佳尺寸，快速傅立叶变换要求尺寸为2的n次方
    Mat padded; //将输入图像延扩到最佳的尺寸  在边缘添加0
    // BORDER_CONSTANT: 常量 （xxx|abcd|xxx） 填充边界（0填充）
    copyMakeBorder(src, padded, 0, h - src.rows, 0, w - src.cols, BORDER_CONSTANT, Scalar::all(0));
    padded.convertTo(padded, CV_32FC1);
    return padded;
}

//频率域滤波
Mat frequency_filter(Mat &src, Mat &blur)
{
    //***********************DFT*******************
    Mat plane[] = { src, Mat::zeros(src.size() , CV_32FC1) }; //创建通道，存储dft后的实部与虚部（CV_32F，必须为单通道数）
    Mat complexIm;
    merge(plane, 2, complexIm);//合并通道 （把两个矩阵合并为一个2通道的Mat类容器） ,为延扩后的图像增添一个初始化为0的通道
    dft(complexIm, complexIm);//进行傅立叶变换，结果保存在自身

    //***************中心化********************
    split(complexIm, plane);//分离通道（数组分离）
//    plane[0] = plane[0](Rect(0, 0, plane[0].cols & -2, plane[0].rows & -2));//这里为什么&上-2具体查看opencv文档
//    //其实是为了把行和列变成偶数 -2的二进制是11111111.......10 最后一位是0
    int cx = plane[0].cols / 2; int cy = plane[0].rows / 2;//以下的操作是移动图像  (零频移到中心)
    Mat part1_r(plane[0], Rect(0, 0, cx, cy));//元素坐标表示为(cx,cy)
    Mat part2_r(plane[0], Rect(cx, 0, cx, cy));
    Mat part3_r(plane[0], Rect(0, cy, cx, cy));
    Mat part4_r(plane[0], Rect(cx, cy, cx, cy));

    Mat temp;
    part1_r.copyTo(temp);//左上与右下交换位置(实部)
    part4_r.copyTo(part1_r);
    temp.copyTo(part4_r);

    part2_r.copyTo(temp);//右上与左下交换位置(实部)
    part3_r.copyTo(part2_r);
    temp.copyTo(part3_r);

    Mat part1_i(plane[1], Rect(0, 0, cx, cy));//元素坐标(cx,cy)
    Mat part2_i(plane[1], Rect(cx, 0, cx, cy));
    Mat part3_i(plane[1], Rect(0, cy, cx, cy));
    Mat part4_i(plane[1], Rect(cx, cy, cx, cy));

    part1_i.copyTo(temp);//左上与右下交换位置(虚部)
    part4_i.copyTo(part1_i);
    temp.copyTo(part4_i);

    part2_i.copyTo(temp);//右上与左下交换位置(虚部)
    part3_i.copyTo(part2_i);
    temp.copyTo(part3_i);

    //*****************滤波器函数与DFT结果的乘积****************
    Mat blur_r, blur_i, BLUR;
    multiply(plane[0], blur, blur_r); //滤波（实部与滤波器模板对应元素相乘）
    multiply(plane[1], blur, blur_i);//滤波（虚部与滤波器模板对应元素相乘）
    Mat plane1[] = { blur_r, blur_i };
    merge(plane1, 2, BLUR);//实部与虚部合并

      //*********************得到原图频谱图***********************************
    magnitude(plane[0], plane[1], plane[0]);//获取幅度图像，0通道为实部通道，1为虚部，因为二维傅立叶变换结果是复数
    plane[0] += Scalar::all(1);//傅立叶变换后的图片不好分析，进行对数处理，结果比较好看
    log(plane[0], plane[0]);// float型的灰度空间为[0，1])
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);//归一化便于显示

    idft(BLUR, BLUR);//idft结果也为复数
    split(BLUR, plane);//分离通道，主要获取通道
    magnitude(plane[0], plane[1], plane[0]);//求幅值(模)
    normalize(plane[0], plane[0], 1, 0, CV_MINMAX);//归一化便于显示
    return plane[0];//返回参数
}

/*
保留低频信号（低通滤波）,低频信号对应图像的中心部分
*/
void MainWindow::func3_lowpass(float sigma) {
//    cv::Mat mat = QImage2Mat(leftImg3);
//    cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
//    cv::Mat result = ideal_low_pass_filter(mat, sigma);
//    result = result(cv::Rect(0, 0, mat.cols, mat.rows));
//    result.convertTo(result, CV_8UC1);
//    rightImg3 = Mat2QImage(result);

    Mat image, image_gray, image_output, image_transform;   //定义输入图像，灰度图像，输出图像
    image = QImage2Mat(leftImg3); cvtColor(image, image_gray, COLOR_BGR2GRAY);

    //1、傅里叶变换，image_output为可显示的频谱图，image_transform为傅里叶变换的复数结果
    My_DFT(image_gray, image_output, image_transform);
    imshow("频谱图", image_output);

    //2、理想低通滤波
    // Mat_: 偏特化版本，这里图像要转为float类型
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);//分离通道，获取实部虚部
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;//频谱图中心坐标
    int core_y = image_transform_real.cols / 2;

    float d0 = sigma;
    for (int i = 0; i < image_transform_real.rows; i++)
    {
        for (int j = 0; j < image_transform_real.cols; j++)
        {
            double d = sqrt(pow(i - core_x, 2) + pow(j - core_y, 2));
            if (d > d0) {
                image_transform_real.at<float>(i, j) = 0;
                image_transform_imag.at<float>(i, j) = 0;
            }
        }
    }
    planes[0] = image_transform_real;
    planes[1] = image_transform_imag;
    Mat image_transform_ilpf;//定义理想低通滤波矩阵
    merge(planes, 2, image_transform_ilpf);

    //3、傅里叶逆变换
    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);//傅立叶逆变换
    split(image_transform_ilpf, iDft);//分离通道，主要获取0通道
    magnitude(iDft[0], iDft[1], iDft[0]); //计算复数的幅值，保存在iDft[0]
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);//归一化处理

    imshow("理想低通滤波", iDft[0]);//显示逆变换图像
    waitKey(0);

//    qDebug() << F32result.type() << "++++++++++++++++++++++++++++++++==\n"; // 5 -> 32F
//    qDebug() << F32result.channels() << "++++++++++++++++++++++++++++++++==\n"; // 1
    /*
     *
     Mat F32result;
    (iDft[0]).copyTo(F32result);
    Mat U8result(F32result.rows, F32result.cols, CV_8UC1);
    for (int i = 0; i < F32result.rows; i++) {
        for (int j = 0; j < F32result.cols; j++) {
            U8result.at<uchar>(i, j) = qBound(0.0, 1.0 * F32result.at<float>(i, j), 255.0);
        }
        qDebug() << "\n";
    }
    */
//    F32result.convertTo(U8result, CV_8U, 255.0);
    // src.convertTo(dst, CV_8UC1)这个函数，只能进行depth的转换，不能转换通道。
    // 要改变通道数，要使用 cv::cvtColor(src, dst, COLOR_BGR2GRAY);   3通道就转化成了单通道
}
/*
保留高频信号（高通滤波）,高频信号对应图像的边缘部分
*/
void MainWindow::func3_highpass(float sigma) {
    Mat image, image_gray, image_output, image_transform;
    image = QImage2Mat(leftImg3); cvtColor(image, image_gray, COLOR_BGR2GRAY);

    My_DFT(image_gray, image_output, image_transform);
    imshow("频谱图", image_output);
    Mat planes[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    split(image_transform, planes);
    Mat image_transform_real = planes[0];
    Mat image_transform_imag = planes[1];

    int core_x = image_transform_real.rows / 2;
    int core_y = image_transform_real.cols / 2;

    float d0 = sigma;
    for (int i = 0; i < image_transform_real.rows; i++)
    {
        for (int j = 0; j < image_transform_real.cols; j++)
        {
            double d = sqrt(pow(i - core_x, 2) + pow(j - core_y, 2));
            if (d <= d0) {
                image_transform_real.at<float>(i, j) = 0;
                image_transform_imag.at<float>(i, j) = 0;
            }
        }
    }
    planes[0] = image_transform_real;
    planes[1] = image_transform_imag;
    Mat image_transform_ilpf;
    merge(planes, 2, image_transform_ilpf);

    Mat iDft[] = { Mat_<float>(image_output), Mat::zeros(image_output.size(),CV_32F) };
    idft(image_transform_ilpf, image_transform_ilpf);
    split(image_transform_ilpf, iDft);
    magnitude(iDft[0], iDft[1], iDft[0]);
    normalize(iDft[0], iDft[0], 0, 1, NORM_MINMAX);

    imshow("理想高通滤波", iDft[0]);
    waitKey(0);
}


void MainWindow::func3_gaussian_lowpass(float sigma) {
    Mat image = QImage2Mat(leftImg3); cvtColor(image, image, COLOR_BGR2GRAY);
    Mat padded = image_make_border(image);

    Mat gaussianBlur(padded.size(), CV_32FC1);
    float d0 = 2 * sigma*sigma;//高斯函数参数，越小，频率高斯滤波器越窄，滤除高频成分越多，图像就越平滑
    for (int i = 0; i < padded.rows; i++) {
        for (int j = 0; j < padded.cols; j++) {
            float d = pow(float(i - padded.rows / 2), 2) + pow(float(j - padded.cols / 2), 2);
            gaussianBlur.at<float>(i, j) = expf(-d / d0);
        }
    }
    imshow("高斯低通滤波器", gaussianBlur);

    Mat result = frequency_filter(padded, gaussianBlur);
    result = result(cv::Rect(0, 0, result.cols, result.rows));
    imshow("高斯低通滤波", result);
}

void MainWindow::func3_gaussian_highpass(float sigma) {
    Mat image = QImage2Mat(leftImg3); cvtColor(image, image, COLOR_BGR2GRAY);
    Mat padded = image_make_border(image);

    Mat gaussianBlur(padded.size(), CV_32FC1);
    float d0 = 2 * sigma*sigma;
    for (int i = 0; i < padded.rows; i++) {
        for (int j = 0; j < padded.cols; j++) {
            float d = pow(float(i - padded.rows / 2), 2) + pow(float(j - padded.cols / 2), 2);
            gaussianBlur.at<float>(i, j) = 1 - expf(-d / d0);
        }
    }
    imshow("高斯高通滤波器", gaussianBlur);

    Mat result = frequency_filter(padded, gaussianBlur);
    result = result(cv::Rect(0, 0, result.cols, result.rows));
    imshow("高斯高通滤波", result);
}

/**
 *  applyColorMap: 伪彩色图像处理
 */
void MainWindow::func4_pseudocolor() {
    QImage image = mainImg4.copy();
    Mat gray = QImage2Mat(image); cvtColor(gray, gray, COLOR_BGR2GRAY);

//    Mat pseudo1; applyColorMap(gray, pseudo1, COLORMAP_HOT);
    Mat pseudo2; applyColorMap(gray, pseudo2, COLORMAP_PINK);
//    Mat pseudo3; applyColorMap(gray, pseudo3, COLORMAP_RAINBOW);
//    Mat pseudo4; applyColorMap(gray, pseudo4, COLORMAP_HSV);
    Mat pseudo5; applyColorMap(gray, pseudo5, COLORMAP_TURBO);

    cv::resize(pseudo2, pseudo2, Size(ui->label_main_image4->width(), ui->label_main_image4->height()));
    imshow("COLORMAP_PINK", pseudo2);
    cv::resize(pseudo5, pseudo5, Size(ui->label_main_image4->width(), ui->label_main_image4->height()));
    imshow("COLORMAP_TURBO", pseudo5);
    waitKey(0);

//    Mat dst;
//    for (int i = 0; i < 22; i++) {
//        applyColorMap(gray, dst, i);
//        cv::resize(dst, dst, Size(ui->label_main_image4->width(), ui->label_main_image4->height()));
//        imshow("map"+std::to_string(i)+".jpg", dst);
//        waitKey(500);
//    }
}

/*      边缘检测
    使用Canny边缘检测算法来实现边缘检测。Canny算法可以用来检测图像中的边缘，并且可以提高边缘检测的精度。
    要使用Canny算法，首先需要将原图像转换为灰度图像，然后应用高斯模糊来消除噪声。
    接下来，应用Sobel算子来检测图像的水平和垂直梯度，并计算每个像素点的梯度幅值和方向。
    最后，通过非极大值抑制和双阈值检测来确定哪些像素点为边缘像素，并将它们高亮显示
        这里只用了对角相减，4邻域
*/
void MainWindow::func4_edge() {
//    edgeImg = (originImg.copy()).convertToFormat(QImage::Format_ARGB32);
    QImage edgeImg = mainImg4.copy();
    QColor color0, color1, color2, color3;
    int r = 0, g = 0, b = 0, rgb = 0;
    int r1 = 0, g1 = 0, b1 = 0, rgb1 = 0;
    int a = 0;

	// (1 1)
	// (-1 -1)
    for(int y = 0; y < edgeImg.height() - 1; y++) {
        for(int x = 0; x < edgeImg.width() - 1; x++) {
            color0 =   QColor (edgeImg.pixel(x,y));
            color1 =   QColor (edgeImg.pixel(x + 1,y));
            color2 =   QColor (edgeImg.pixel(x,y + 1));
            color3 =   QColor (edgeImg.pixel(x + 1,y + 1));
            r = abs(color0.red() - color3.red());
            g = abs(color0.green() - color3.green());
            b = abs(color0.blue() - color3.blue());
            rgb = r + g + b;

            r1 = abs(color1.red() - color2.red());
            g1= abs(color1.green() - color2.green());
            b1 = abs(color1.blue() - color2.blue());
            rgb1 = r1 + g1 + b1;

            a = rgb + rgb1;
            a = a > 255 ? 255 : a;

            edgeImg.setPixel(x,y,qRgb(a,a,a));
        }
    }
    ui->label_main_image4->setPixmap(QPixmap::fromImage(edgeImg));
}

/*
    模糊处理通常是对灰度图像进行处理。灰度图像只包含黑白两种颜色，每个像素都有一个灰度值，表示该像素的亮度。这样的图像更容易处理，也更容易计算像素与周围像素的差异。
    但是，也可以对彩色图像进行模糊处理。例如，可以对图像的每个颜色通道分别进行模糊处理，然后再将处理后的颜色通道合并起来，形成最终的彩色图像。这样的处理方式也可以有效地减少噪声并提高图像质量
*/
//void MainWindow::func1_blur() {
//    blurImg = originImg.copy();
//    blurImg.scaled(ui->label_blur_image->size(), Qt::IgnoreAspectRatio);

//    QImage scaleImg = blurImg.scaled(blurImg.width() / 3, blurImg.height() / 3);
//    // 分别用于指定图像缩放时的对齐方式和平滑模式
//    // Qt::SmoothTransformation 则指定使用平滑模式，在缩放图像时，会使用双线性插值算法，来保证图像的质量
//    blurImg = scaleImg.scaled(blurImg.width(), blurImg.height(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

//    ui->label_blur_image->setPixmap(QPixmap::fromImage(blurImg));
//}

// 差分运算来增强图像的边缘信息，然后将差分后的像素值设置到新的图像中
void MainWindow::func2_difference(QImage &image) {
    // 创建一个新的QImage图像，用于存储差分运算后的结果
    QImage result = QImage(image.width(), image.height(), image.format());
    // 对每个像素进行差分运算
    for (int y = 0; y < image.height(); y++) {
        for (int x = 0; x < image.width(); x++) {
            // 计算差分后的像素值
            int red = 0, green = 0, blue = 0, alpha = 0;
            // 计算与相邻像素的差值
            if (x > 0) {
                QRgb leftPixel = image.pixel(x - 1, y);
                red = qAbs(qRed(leftPixel) - qRed(image.pixel(x, y)));
                green = qAbs(qGreen(leftPixel) - qGreen(image.pixel(x, y)));
                blue = qAbs(qBlue(leftPixel) - qBlue(image.pixel(x, y)));
                alpha = qAbs(qAlpha(leftPixel) - qAlpha(image.pixel(x, y)));
            }
            if (y > 0) {
                QRgb topPixel = image.pixel(x, y - 1);
                red = qAbs(qRed(topPixel) - qRed(image.pixel(x, y)));
                green = qAbs(qGreen(topPixel) - qGreen(image.pixel(x, y)));
                blue = qAbs(qBlue(topPixel) - qBlue(image.pixel(x, y)));
                alpha = qAbs(qAlpha(topPixel) - qAlpha(image.pixel(x, y)));
            }
            // 设置差分后的像素值
            result.setPixel(x, y, qRgba(red, green, blue, alpha));
        }
    }
    // 保存差分后的结果
    image = result;
}

/**
 * @brief MainWindow::func2_hysteresisThreshold
 * @param image
 * @param lowThreshold
 * @param highThreshold
 */
/*
    1. 对灰度图像进行阈值分割，使用两个阈值来将图像分成三个部分：低于下阈值的像素、在两个阈值之间的像素和高于上阈值的像素。
    2. 对于低于下阈值的像素，将它们设为黑色。
    3. 对于在两个阈值之间的像素，检查它们周围的像素。如果有一个像素高于上阈值，那么将该像素设为白色，否则设为黑色。
    4. 对于高于上阈值的像素，将它们设为白色。
*/
void MainWindow::func2_hysteresisThreshold(QImage& image, int lowThreshold, int highThreshold) {
    // 将原始图像转换为灰度图像
    QImage grayImage = image.convertToFormat(QImage::Format_Grayscale8);
    // 创建一个新的图像，用于保存处理后的图像
    QImage resultImage(grayImage.width(), grayImage.height(), QImage::Format_Grayscale8);
   // 遍历每一个像素
   for (int y = 0; y < grayImage.height(); y++) {
       for (int x = 0; x < grayImage.width(); x++) {
           // 获取像素值
           int pixelValue = qGray(grayImage.pixel(x, y));
           // 对像素进行阈值分割
           if (pixelValue < lowThreshold) {
               // 将低于下阈值的像素设为黑色
               resultImage.setPixel(x, y, qRgb(0, 0, 0));
           } else if (pixelValue >= lowThreshold && pixelValue <= highThreshold) {
               // 对于在两个阈值之间的像素，检查它们周围的像素
               bool isWhite = false;
               for (int dy = -1; dy <= 1; dy++) {
                   for (int dx = -1; dx <= 1; dx++) {
                       // 跳过当前像素本身
                       if (dx == 0 && dy == 0) continue;

                       // 计算像素的坐标
                       int nx = x + dx;
                       int ny = y + dy;

                       // 判断像素是否在图像范围内
                       if (nx < 0 || nx >= grayImage.width() || ny < 0 || ny >= grayImage.height()) continue;

                       // 获取像素值
                       int neighborValue = qGray(grayImage.pixel(nx, ny));

                       // 如果周围有一个像素高于上阈值，那么该像素设为白色
                       if (neighborValue > highThreshold) {
                           isWhite = true;
                           break;
                       }
                   }

                   if (isWhite) break;
               }

               if (isWhite) {
                   resultImage.setPixel(x, y, qRgb(255, 255, 255));
               } else {
                   resultImage.setPixel(x, y, qRgb(0, 0, 0));
               }
           } else if (pixelValue > highThreshold) {
               // 将高于上阈值的像素设为白色
               resultImage.setPixel(x, y, qRgb(255, 255, 255));
           }
       }
   }
   image = resultImage;
}


/////////////////////////////////////////////////////////////// SLOTS ////////////////////////////////////////////////////////////////

void MainWindow::on_OpenImg() {
    QString OpenFile, OpenFilePath;
    QImage image;
    OpenFile = QFileDialog::getOpenFileName(this,
                                            tr("Please choose an image"),
                                            "D:/",
                                            "Image files(*.jpg *.png)"); // ;;All files(*.*)
    if (OpenFile != "") {
        if (image.load(OpenFile)) {
            int page = ui->tabWidget->currentIndex(); // 我觉得每次重开一张图片，应该把之前的清空了
            if (page == 0) {
                originImg = image.scaled(ui->label_origin_image->size(), Qt::IgnoreAspectRatio); // 图片适应label
    //            ui->label_origin_image->setScaledContents(true); // label适应图片 //好像反了，但记住这两种用法即可。
                ui->label_origin_image->setPixmap(QPixmap::fromImage(originImg));
                this->grayImg = (this->originImg.copy()).convertToFormat(QImage::Format_Grayscale8);
                tab[page].clear();
                tab[page].append({ originImg, 0 });
            } else if (page == 1) {
                tab[page].clear();
//                mainImg = image.scaled(ui->label_main_image->size(), Qt::IgnoreAspectRatio);
//                ui->label_main_image->setPixmap(QPixmap::fromImage(mainImg));

                // 这里要生成所有的图片
//                cursor = 1, last_cursor = 0;
                mainImg = (image.copy()).convertToFormat(QImage::Format_ARGB32_Premultiplied); tab[page].append({ mainImg, 0 });
                func2_mean(); tab[page].append({ mainImg, 1 });
                func2_median(); tab[page].append({ mainImg, 2 });
                func2_gaussian(); tab[page].append({ mainImg, 3 });
                func2_bilateral(); tab[page].append({ mainImg, 4 });
                func2_laplacian(); tab[page].append({ mainImg, 5 });
                func2_max(); tab[page].append({ mainImg, 6 });
                func2_sobel(); tab[page].append({ mainImg, 7 });

                emit signal_combox2(cursor);

                tab[-1].clear();
                for (int i = 0; i < 8; i++) {
                    tab[-1].append({ tab[page][i].img.copy(), i });
                }
            } else if (page == 2) {
                tab[page].clear();
                leftImg3 = (image.copy()).convertToFormat(QImage::Format_RGB888);
                tab[page].append({ leftImg3, 0 });
                ui->label_origin_image1->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
                func3_pepper(); tab[page].append({ leftImg3, 1 });

                leftImg3 = tab[page].at(0).img; // 置回原图
            } else if (page == 3) {
                tab[page].clear();
                mainImg4 = (image.copy()).convertToFormat(QImage::Format_RGB888);
                tab[page].append({ mainImg4, 0 });
                ui->label_main_image4->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
            }
        }
    }
    QFileInfo OpenFileInfo;
    OpenFileInfo = QFileInfo(OpenFile);
    OpenFilePath = OpenFileInfo.filePath();
    originPath = OpenFilePath; // 设置好当前源图像的路径
    ui->statusbar->showMessage(OpenFilePath, 8000);
}

void MainWindow::on_MusicPlay() {
    // 使用QtConcurrent::run()函数异步执行
    if (musicPlayer->state() == QMediaPlayer::PlayingState) {
        QtConcurrent::run([&](){ musicPlayer->pause(); });
    } else {
        QtConcurrent::run([&](){ musicPlayer->play(); });
    }
}

void MainWindow::on_showOringinImg() {
    int page = ui->tabWidget->currentIndex();
    if (page == 0) {
        auto &it = tab[page];
        for (auto &var: it) {
            int label_idx = var.label_idx;
            QImage &img = var.img;
            LabelIdxToSetPixmap(img, label_idx);
        }
    } else if (page == 1) {
//        ui->label_main_image->setPixmap(QPixmap::fromImage(tab[page].at(this->cursor).img));
        tab[page].clear();
        for (int i = 0; i < 8; i++) {
            tab[page].append({ tab[-1][i].img.copy(), i });
        }
        emit signal_combox2(this->cursor);
    } else if (page == 2) {
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
    } else if (page == 3) {
        ui->label_main_image4->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
    }
}

void MainWindow::on_showGrayImg() {
    int page = ui->tabWidget->currentIndex();
    if (page == 0) {
        ui->label_origin_image->setPixmap(QPixmap::fromImage(grayImg));
    } else if (page == 1) {
        QImage now = tab[page].at(this->cursor).img;
        QImage tmp = now.copy();
        now = now.convertToFormat(QImage::Format_Grayscale8);
        tab[page][this->cursor] = { now, this->cursor };
        emit signal_combox2(this->cursor);
        tab[page][this->cursor] = { tmp, this->cursor };
    } else if (page == 2) {
        QImage now = tab[page].at(0).img.convertToFormat(QImage::Format_Grayscale8);
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(now));
    } else if (page == 3) {
        QImage now = tab[page].at(0).img.convertToFormat(QImage::Format_Grayscale8);
        ui->label_main_image4->setPixmap(QPixmap::fromImage(now));
    }
}

void MainWindow::on_saveImg() {
    int page = ui->tabWidget->currentIndex();
    imwrite("./pic/colorImg.jpg", QImage2Mat(tab[page].at(0).img));
}

void MainWindow::on_ComBox1(int index) {
    if (index == 1) {
        this->func1_imadjust();
        tab[0].append({ imadjustImg, 1 });
        this->showHistogram(this->imadjustImg);
    } else if (index == 2) {
        this->func1_binary();
        tab[0].append({ binaryImg, 2 });
        this->showHistogram(this->binaryImg);
    } else if (index == 3) {
        this->func1_gamma();
        tab[0].append({ gammaImg, 3 });
        this->showHistogram(this->gammaImg);
    } else if (index == 4) {
        this->func1_grayavg();
        tab[0].append({ grayavgImg, 4 });
        this->showHistogram(this->grayavgImg);
    } else if (index == 5) {
        this->func1_log();
        tab[0].append({ logImg, 5 });
        this->showHistogram(this->logImg);
    } else if (index == 6) {
        this->func1_power();
        tab[0].append({ powerImg, 6 });
        this->showHistogram(this->powerImg);
    } else if (index == 7) {
        this->func1_reverse();
        tab[0].append({ reverseImg, 7 });
        this->showHistogram(this->reverseImg);
    } else if (index == 8) {
        for (int i = 1; i <= 7; i++) {
            ui->comboBox->setCurrentIndex(i);
        }
    }
}

void MainWindow::on_ComBox2(int index) {
    if (index < 1 || index > 7) return ;
    int page = ui->tabWidget->currentIndex();
    cursor = index;
//    if (cursor == last_cursor) return ; -- 这先注释，适配一下on_showGrayImg(); 中的 emit signal_combox2();
//    ui->comboBox2->setCurrentIndex(index); // 本意想改变box2的显示，但是这里会递归触发的！！！
    if (index == 1) {
        ui->label_main_image->setPixmap(QPixmap::fromImage(tab[page].at(1).img));
        ui->label_left_image->setPixmap(QPixmap::fromImage(tab[page].at(1).img));
        ui->label_center_image->setPixmap(QPixmap::fromImage(tab[page].at(2).img));
        ui->label_right_image->setPixmap(QPixmap::fromImage(tab[page].at(3).img));
        ui->widget_left->setStyleSheet(QString::fromUtf8("QWidget#widget_left{border: 3px solid red; border-radius: 8px;}"));
    } else if (index == 7) {
        ui->label_main_image->setPixmap(QPixmap::fromImage(tab[page].at(7).img));
        ui->label_left_image->setPixmap(QPixmap::fromImage(tab[page].at(5).img));
        ui->label_center_image->setPixmap(QPixmap::fromImage(tab[page].at(6).img));
        ui->label_right_image->setPixmap(QPixmap::fromImage(tab[page].at(7).img));
        ui->widget_right->setStyleSheet(QString::fromUtf8("QWidget#widget_right{border: 3px solid red; border-radius: 8px;}"));
    } else {
        ui->label_main_image->setPixmap(QPixmap::fromImage(tab[page].at(index).img));
        ui->label_left_image->setPixmap(QPixmap::fromImage(tab[page].at(index - 1).img));
        ui->label_center_image->setPixmap(QPixmap::fromImage(tab[page].at(index).img));
        ui->label_right_image->setPixmap(QPixmap::fromImage(tab[page].at(index + 1).img));
        ui->widget_center->setStyleSheet(QString::fromUtf8("QWidget#widget_center{border: 3px solid red; border-radius: 8px;}"));
    }
    ui->label_main_name->setText(QString(mainImg_name[index]));
    if (last_cursor == 1) {
        ui->widget_left->setStyleSheet(QString::fromUtf8("QWidget#widget_left{border: 3px solid block; border-radius: 8px;}"));
    } else if (last_cursor == 7) {
        ui->widget_right->setStyleSheet(QString::fromUtf8("QWidget#widget_right{border: 3px solid block; border-radius: 8px;}"));
    } else {
        if (cursor == 1 || cursor == 7)
            ui->widget_center->setStyleSheet(QString::fromUtf8("QWidget#widget_center{border: 3px solid block; border-radius: 8px;}"));
    }
    last_cursor = cursor;
    mainImg = tab[page].at(index).img;
}

void MainWindow::on_ComBox3(int index) {
    int page = ui->tabWidget->currentIndex();
    if (index == 0) {
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
        leftImg3 = tab[page].at(0).img;
        ui->label_origin_name->setText("原图像");
    } else if (index == 1) {
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(tab[page].at(1).img));
        leftImg3 = tab[page].at(1).img;
        ui->label_origin_name->setText("椒盐噪声");
    } else if (index == 2) {
        func3_lowpass();
    } else if (index == 3) {
        func3_highpass();
    } else if (index == 4) {
        func3_gaussian_lowpass();
    } else if (index == 5) {
        func3_gaussian_highpass();
    }
}

void MainWindow::on_ComBox4(int index) {
    int page = ui->tabWidget->currentIndex();
    if (index == 0) {
        ui->label_main_image4->setPixmap(QPixmap::fromImage(tab[page].at(0).img));
    } else if (index == 1) {
        func4_pseudocolor();
    } else if (index == 2) {
        func4_edge();
    }
}

void MainWindow::on_bright_valueChanged(int value) {
    int page = ui->tabWidget->currentIndex();
    int delta = value - lastBright;
    lastBright = value;

    auto bright_change = [=](QImage img) -> QImage {
        if (!img.isGrayscale()) { // aplha表示（0透明~255不透明）
            // 彩色图
            for (int y = 0; y < img.height(); y++) {
                for (int x = 0; x < img.width(); x++) {
                    QRgb color = img.pixel(x, y);
                    int r = qRed(color);
                    int g = qGreen(color);
                    int b = qBlue(color);
                    r = qBound(0, r + delta, 255);
                    g = qBound(0, g + delta, 255);
                    b = qBound(0, b + delta, 255);
                    color = qRgb(r, g, b);
                    img.setPixel(x, y, color);
                }
            }
        } else {
            //非彩色图
            for (int y = 0; y < img.height(); y++) {
                for (int x = 0; x < img.width(); x++) {
                    QRgb color = img.pixel(x, y);
                    int r = qRed(color);
                    r = qBound(0, r + delta, 255);
                    color = qRgb(r, r, r);
                    img.setPixel(x, y, color);
                }
            }
        }
        return img;
    };

    if (page == 0) {
        auto &it = tab[page];
        for (auto &var: it) {
            int label_idx = var.label_idx;
            QImage img = (var.img).copy(); // 这里需要拷贝一份进行修改，而不能直接对tab中的
            LabelIdxToSetPixmap(bright_change(img), label_idx);
        }
    } else if (page == 1) {
        QImage now = bright_change(mainImg);
        tab[page][cursor] = { now, cursor };
        emit signal_combox2(cursor);
    } else if (page == 2) {
        leftImg3 = bright_change(leftImg3);
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(leftImg3));
    } else if (page == 3) {
        mainImg4 = bright_change(mainImg4);
        ui->label_main_image4->setPixmap(QPixmap::fromImage(mainImg4));
    }
}

//二值化滑动条
void MainWindow::on_binary_valueChanged(int value) {
    this->func1_binary(value);
}

void MainWindow::on_R_valueChanged(int value) {
    int page = ui->tabWidget->currentIndex();
    int delta = value - lastR;
    lastR = value;

    auto R_change = [=](QImage img) -> QImage {
        // 彩色图
        for (int y = 0; y < img.height(); y++) {
            for (int x = 0; x < img.width(); x++) {
                QRgb color = img.pixel(x, y);
                int r = qRed(color);
                int g = qGreen(color);
                int b = qBlue(color);
                r = qBound(0, r + delta, 255);
                color = qRgb(r, g, b);
                img.setPixel(x, y, color);
            }
        }
        return img;
    };

    if (page == 0) {
        auto &it = tab[page];
        for (auto &var: it) {
            int label_idx = var.label_idx;
            QImage img = (var.img).copy(); // 这里需要拷贝一份进行修改，而不能直接对tab中的
            if (img.format() == QImage::Format_RGB32 || img.format() == QImage::Format_ARGB32 \
                    || img.format() == QImage::Format_ARGB32_Premultiplied) { // aplha表示（0透明~255不透明）
//                // 彩色图
//                for (int y = 0; y < img.height(); y++) {
//                    for (int x = 0; x < img.width(); x++) {
//                        QRgb color = img.pixel(x, y);
//                        int r = qRed(color);
//                        int g = qGreen(color);
//                        int b = qBlue(color);
//                        r = qBound(0, r + delta, 255);
//                        color = qRgb(r, g, b);
//                        img.setPixel(x, y, color);
//                    }
//                }
                LabelIdxToSetPixmap(R_change(img), label_idx);
            }
            else { /*非彩色图*/ }
        }
    } else if (page == 1) {
        QImage now = R_change(mainImg);
        tab[page][cursor] = { now, cursor };
        emit signal_combox2(cursor);
    } else if (page == 2) {
        leftImg3 = R_change(leftImg3);
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(leftImg3));
    } else if (page == 3) {
        mainImg4 = R_change(mainImg4);
        ui->label_main_image4->setPixmap(QPixmap::fromImage(mainImg4));
    }
}

void MainWindow::on_G_valueChanged(int value) {
    int page = ui->tabWidget->currentIndex();
    int delta = value - lastG;
    lastG = value;

    auto G_change = [=](QImage img) -> QImage {
        // 彩色图
        for (int y = 0; y < img.height(); y++) {
            for (int x = 0; x < img.width(); x++) {
                QRgb color = img.pixel(x, y);
                int r = qRed(color);
                int g = qGreen(color);
                int b = qBlue(color);
                g = qBound(0, g + delta, 255);
                color = qRgb(r, g, b);
                img.setPixel(x, y, color);
            }
        }
        return img;
    };

    if (page == 0) {
        auto &it = tab[page];
        for (auto &var: it) {
            int label_idx = var.label_idx;
            QImage img = (var.img).copy(); // 这里需要拷贝一份进行修改，而不能直接对tab中的
            if (img.format() == QImage::Format_RGB32 || img.format() == QImage::Format_ARGB32 \
                    || img.format() == QImage::Format_ARGB32_Premultiplied) { // aplha表示（0透明~255不透明）{
                LabelIdxToSetPixmap(G_change(img), label_idx);
            }
            else { /*非彩色图*/ }
        }
    } else if (page == 1) {
        QImage now = G_change(mainImg);
        tab[page][cursor] = { now, cursor };
        emit signal_combox2(cursor);
    } else if (page == 2) {
        leftImg3 = G_change(leftImg3);
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(leftImg3));
    } else if (page == 3) {
        mainImg4 = G_change(mainImg4);
        ui->label_main_image4->setPixmap(QPixmap::fromImage(mainImg4));
    }
}

void MainWindow::on_B_valueChanged(int value) {
    int page = ui->tabWidget->currentIndex();
    int delta = value - lastB;
    lastB = value;

    auto B_change = [=](QImage img) -> QImage {
        // 彩色图
        for (int y = 0; y < img.height(); y++) {
            for (int x = 0; x < img.width(); x++) {
                QRgb color = img.pixel(x, y);
                int r = qRed(color);
                int g = qGreen(color);
                int b = qBlue(color);
                b = qBound(0, b + delta, 255);
                color = qRgb(r, g, b);
                img.setPixel(x, y, color);
            }
        }
        return img;
    };

    if (page == 0) {
        auto &it = tab[page];
        for (auto &var: it) {
            int label_idx = var.label_idx;
            QImage img = (var.img).copy(); // 这里需要拷贝一份进行修改，而不能直接对tab中的
            if (img.format() == QImage::Format_RGB32 || img.format() == QImage::Format_ARGB32 \
                    || img.format() == QImage::Format_ARGB32_Premultiplied) { // aplha表示（0透明~255不透明） {
                LabelIdxToSetPixmap(B_change(img), label_idx);
            }
            else { /*非彩色图*/ }
        }
    } else if (page == 1) {
        QImage now = B_change(mainImg);
        tab[page][cursor] = { now, cursor };
        emit signal_combox2(cursor);
    } else if (page == 2) {
        leftImg3 = B_change(leftImg3);
        ui->label_origin_image1->setPixmap(QPixmap::fromImage(leftImg3));
    } else if (page == 3) {
        mainImg4 = B_change(mainImg4);
        ui->label_main_image4->setPixmap(QPixmap::fromImage(mainImg4));
    }
}

void MainWindow::on_updateDateTime() {
    ui->statusbar->showMessage((QDateTime::currentDateTime()).toString("yyyy-MM-dd hh:mm:ss"));
}

//////////////////////////////////////////// Setting ///////////////////////////////////////////////////////////////////////////

/*
void MainWindow::fft(Complex* data, int n) {
    if (n == 1)
        return;
    Complex wn = exp(Complex(0, -2 * M_PI / n));
    Complex w = 1;

    Complex* even = new Complex[n / 2];
    Complex* odd = new Complex[n / 2];
    for (int i = 0; i < n / 2; i++)
    {
        even[i] = data[i * 2];
        odd[i] = data[i * 2 + 1];
    }

    fft(even, n / 2);
    fft(odd, n / 2);

    for (int k = 0; k < n / 2; k++)
    {
        data[k] = even[k] + w * odd[k];
        data[k + n / 2] = even[k] - w * odd[k];
        w = w * wn;
    }

    delete[] even;
    delete[] odd;
}
*/

/**
 * 实现了 FFT 变换的过程，并将结果保存到一个新的图像文件中。
 * 这里使用了振幅谱图的简单变换方法，将复数的实部的绝对值映射到图像的像素值。
 * 这里的 FFT 实现是基于递归的 Cooley-Tukey 算法，时间复杂度为 O(n log n)。这个算法在处理大型信号时可能会很慢
 */
/*
void MainWindow::fft(QImage& image) {
    // 确保图像是灰度图像
    if (image.format() != QImage::Format_Grayscale8)
        image = image.convertToFormat(QImage::Format_Grayscale8);

    int width = image.width();
    int height = image.height();

    // 将图像数据复制到一个复数数组中
    Complex* data = new Complex[width * height];
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            data[y * width + x] = image.pixelColor(x, y).value();
        }
    }
    // 对每一行和每一列分别进行 FFT 变换
    for (int y = 0; y < height; y++)
        fft(data + y * width, width);
    for (int x = 0; x < width; x++)
    {
        Complex* col = new Complex[height];
        for (int y = 0; y < height; y++)
            col[y] = data[y * width + x];
        fft(col, height);
        for (int y = 0; y < height; y++)
            data[y * width + x] = col[y];
        delete[] col;
    }

    // 将 FFT 变换后的结果复制回图像
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // 这里使用了振幅谱图的简单变换方法
            image.setPixelColor(x, y, QColor::fromRgb(qAbs(data[y * width + x].real())));
        }
    }

    delete[] data;
}
*/

//Mat转图像
QImage MainWindow::Mat2QImage(const cv::Mat& mat)
{
//    // 8-bits unsigned, NO. OF CHANNELS = 1
//    if (mat.type() == CV_8UC1)
//    {
//        QImage image(mat.cols, mat.rows, QImage::Format_Indexed8);
//        // Set the color table (used to translate colour indexes to qRgb values)
//        image.setColorCount(256);
//        for (int i = 0; i < 256; i++)
//        {
//            image.setColor(i, qRgb(i, i, i));
//        }
//        // Copy input Mat
//        uchar *pSrc = mat.data;
//        for (int row = 0; row < mat.rows; row++)
//        {
//            uchar *pDest = image.scanLine(row);
//            memcpy(pDest, pSrc, mat.cols);
//            pSrc += mat.step;
//        }
//        return image;
//    }
//    // 8-bits unsigned, NO. OF CHANNELS = 3
//    else if (mat.type() == CV_8UC3)
//    {
//        // Copy input Mat
//        const uchar *pSrc = (const uchar*)mat.data;
//        // Create QImage with same dimensions as input Mat
//        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
//        return image.rgbSwapped();
//    }
//    else if (mat.type() == CV_8UC4)
//    {
//        // Copy input Mat
//        const uchar *pSrc = (const uchar*)mat.data;
//        // Create QImage with same dimensions as input Mat
//        QImage image(pSrc, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32);
//        return image.copy();
//    }
//    else
//    {
//        qDebug() << mat.type() << "========================+*******************\n";
//        qDebug() << "cant convert to QImage *******************+=====================\n";
//        return QImage();
//    }
    //判断m的类型，可能是CV_8UC1  CV_8UC2  CV_8UC3  CV_8UC4
    switch(mat.type())
    {
        //QIamge 构造函数, ((const uchar *data, 宽(列),高(行), 一行共多少个（字节）通道，宽度*字节数，宏参数)
        case CV_8UC1:
        {
            QImage img((uchar *)mat.data, mat.cols, mat.rows, mat.cols * 1,QImage::Format_Grayscale8);
            return img;
        }
            break;
        case CV_8UC3:   //一个像素点由三个字节组成
        {
            //cvtColor(m,m,COLOR_BGR2RGB); BGR转RGB
            QImage img((uchar *)mat.data, mat.cols, mat.rows, mat.cols * 3, QImage::Format_RGB888);
            return img.rgbSwapped(); //opencv是BGR  Qt默认是RGB  所以RGB顺序转换
        }
            break;
        case CV_8UC4:
        {
            QImage img((uchar *)mat.data, mat.cols, mat.rows, mat.cols * 4, QImage::Format_RGBA8888);
            return img;
        }
            break;
        default:
        {
            QImage img; //如果遇到一个图片均不属于这三种，返回一个空的图片
            return img;
        }
    }
}

cv::Mat MainWindow::QImage2Mat(const QImage& image)
{
    cv::Mat mat;
    switch (image.format())
    {
        case QImage::Format_ARGB32:
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32_Premultiplied:
            mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void*)image.constBits(), image.bytesPerLine());
            break;
        case QImage::Format_RGB888:
            mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void*)image.constBits(), image.bytesPerLine());
            cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR); // opencv 3.0
            // opencv 2.0 -> CV_RGB2BGR
            break;
        case QImage::Format_Indexed8:
            mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void*)image.constBits(), image.bytesPerLine());
            break;
        default:
            break;
    }
    return mat;
}

void MainWindow::LabelIdxToSetPixmap(QImage img, int label_idx) {
    switch (label_idx) {
        case 0:
            ui->label_origin_image->setPixmap(QPixmap::fromImage(img/*originImg*/)); break;
        case 1:
            ui->label_imadjust_image->setPixmap(QPixmap::fromImage(img/*imadjustImg*/)); break;
        case 2:
            ui->label_binary_image->setPixmap(QPixmap::fromImage(img/*binaryImg*/)); break;
        case 3:
            ui->label_gamma_image->setPixmap(QPixmap::fromImage(img/*gammaImg*/)); break;
        case 4:
            ui->label_grayavg_image->setPixmap(QPixmap::fromImage(img/*grayavgImg*/)); break;
        case 5:
            ui->label_log_image->setPixmap(QPixmap::fromImage(img/*grayavgImg*/)); break;
        case 6:
            ui->label_power_image->setPixmap(QPixmap::fromImage(img/*grayavgImg*/)); break;
        case 7:
            ui->label_reverse_image->setPixmap(QPixmap::fromImage(img/*grayavgImg*/)); break;
    }
}

void MainWindow::Setting() {
    qDebug() << Q_FUNC_INFO << "\n";
//    QMenu *menu1 = new QMenu("文件", this);
//    QMenu *menu2 = new QMenu("编辑", this);
//    QMenu *menu3 = new QMenu("格式", this);
//    ui->menubar->addSeparator();
//    ui->menubar->addMenu(menu1);
//    ui->menubar->addMenu(menu2);
//    ui->menubar->addMenu(menu3);
    setWindowFlags(windowFlags() & ~Qt::WindowMaximizeButtonHint);    // 禁止最大化按钮
    setFixedSize(this->width(), this->height());                      // 禁止拖动窗口大小

    // 先弄好了各个布局，所以这里的设置都不成功了
//    // 由于tab1，我早先弄好了，没设置背景图。在此用代码设置，改为全屏吧。
//    QLabel *label_back_ground1 = new QLabel(this);
//    label_back_ground1->setPixmap(QPixmap(":/src/background.png"));   // :/ 表示qrc /下的文件，歌曲放在debug下，所以 ./即可
//    label_back_ground1->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored); // 设置Label的大小策略为忽略
//    label_back_ground1->setGeometry(0, 0, this->width(), this->height());// 设置Label的位置和大小，使其完全覆盖界面
//    QGraphicsOpacityEffect *effect = new QGraphicsOpacityEffect(label_back_ground1);
//    effect->setOpacity(0.7); // label的设置透明度函数在我版本中没有，那么借助 QGraphicsOpacityEffect 来实现
//    label_back_ground1->lower();

    // Qt5.11.2之后才有 提升+选择底层。 我在这里用代码实现。这是tab2、tab3的背景图
    ui->label_back_ground2->lower();
    ui->label_back_ground3->lower();
    ui->label_back_ground4->lower();


    //////////////////////////////////////////////// 顶部toolBar ////////////////////////////////////////////////

    QAction *New = new QAction(QIcon(":/src/filenew.png"), "新建", this);
    QAction *Open = new QAction(QIcon(":/src/fileopen.png"), "打开", this);
    QAction *Save = new QAction(QIcon(":/src/save.png"), "保存", this);

    //创建toolbar工具条
    ui->toolBar->addAction(New);
    ui->toolBar->addAction(Open);
    ui->toolBar->addAction(Save);
    ui->toolBar->move(0, 25);

    //设置停靠位置
    ui->toolBar->setAllowedAreas(Qt::TopToolBarArea);
    ui->toolBar->setFloatable(false);
    ui->toolBar->setMovable(false);
    //设置文字和图标相对关系
    ui->toolBar->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    ui->toolBar->setOrientation(Qt::Horizontal);
    ui->toolBar->resize(this->geometry().width(),30);


    //////////////////////////////////////////////// 底部stateBar ////////////////////////////////////////////////

//    QLabel *per1 = new QLabel; per1->setText("Qt状态栏"); // 这里左下角就换成显示时间吧。
    QLabel *per2 = new QLabel("Github", this);
    QLabel *per3 = new QLabel;

//    ui->statusbar->addWidget(per1);
    ui->statusbar->addPermanentWidget(per2);
    ui->statusbar->addPermanentWidget(per3); // 显示永久信息
    per2->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
    per2->setText(tr("<a href=\"https://github.com/cauchy-74\">Github</a>"));
    per2->setOpenExternalLinks(true);
//    per3->setFrameStyle(QFrame::Box | QFrame::Sunken); 框框不好看
    per3->setFrameStyle(QFrame::StyledPanel | QFrame::Raised);
    per3->setText(tr("<a href=\"http://blog.csdn.net//qq_52678569\">MyBlog</a>"));
    per3->setOpenExternalLinks(true);//设置可以打开网站链接

    ui->statusbar->setSizeGripEnabled(false);//去掉状态栏右下角的三角

    //////////////////////////////////////////////// 左侧dockwidget ////////////////////////////////////////////////
    /**
     *   Music
     *     1. 改logo的png
     *     2. 导入歌曲
     */
    QImage music_img;
    music_img.load(":/src/music_log.png");
    QPixmap music_pix = QPixmap::fromImage(music_img);
    int w = ui->btn_music->width(), h = ui->btn_music->height();
    QPixmap music_fit = music_pix.scaled(w, h, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
    ui->btn_music->setIcon(QIcon(music_fit));
    ui->btn_music->setIconSize(QSize(80, 80));
    ui->btn_music->setFlat(true);
    ui->btn_music->setStyleSheet("border: 0px"); // 消除边框

//    musicPlayer = new QMediaPlayer;
//    musicPlayer->setMedia(QUrl::fromLocalFile("./src/music.mp3")); // src文件夹放在debug的同级下。
//    musicPlayer->setVolume(50); //0~100音量范围, 默认是100

    //////////////////////////////////////////////// 右侧dockwidget ////////////////////////////////////////////////
    ui->slider_bright->setRange(0, 30);
    ui->slider_bright->setValue(0);
    ui->slider_bright->setTickInterval(1);  // 设置离散

    ui->slider_binary->setRange(0, 255);
    ui->slider_binary->setValue(0);
    ui->slider_binary->setTickInterval(1);

    ui->slider_B->setRange(0, 30);
    ui->slider_B->setValue(0);

    ui->slider_G->setRange(0, 30);
    ui->slider_G->setValue(0);

    ui->slider_R->setRange(0, 30);
    ui->slider_R->setValue(0);

    // tab2
    /*                  预存在这，修改widget的边框颜色
                                // 创建一个QWidget对象
                                QWidget *widget = new QWidget;

                                // 使用palette属性设置widget的边框颜色
                                QPalette palette = widget->palette();
                                palette.setColor(QPalette::Window, Qt::black);
                                widget->setPalette(palette);

                                // 使用styleSheet属性设置widget的边框宽度
                                widget->setStyleSheet("border: 1px solid black");
    */
}
