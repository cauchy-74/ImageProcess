#ifndef MUSICTHREAD_H
#define MUSICTHREAD_H

#include <QUrl>
#include <QThread>
#include <QMediaPlayer>

// 定义线程类
class MusicThread : public QThread {
    Q_OBJECT

public:
    MusicThread() {
        // 创建音乐播放对象
        player = new QMediaPlayer();
        connect(player, &QMediaPlayer::stateChanged, this, &MusicThread::onStateChanged);
    }

    ~MusicThread() {
        // 释放音乐播放对象
        delete player;
    }

    // 设置音乐播放地址
    void setMedia(const QUrl &url) {
        player->setMedia(url);
    }

    void stop() {
        player->stop();
    }
    void play() {
        player->play();
    }

signals:
    // 发送音乐播放状态改变的信号
    void stateChanged(QMediaPlayer::State state);

protected:
    void run() {
        // 播放音乐
//        player->play();
    }

private slots:
    // 处理音乐播放状态改变的信号
    void onStateChanged(QMediaPlayer::State state) {
        // 发送音乐播放状态改变的信号
        emit stateChanged(state);
    }

private:
    QMediaPlayer *player;  // 音乐播放对象
};

#endif // MUSICTHREAD_H
