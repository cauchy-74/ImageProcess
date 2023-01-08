#ifndef MUSICPLAYER_H
#define MUSICPLAYER_H

#include <QtConcurrent>
#include <QMediaPlayer>

class MusicPlayer : public QObject
{
    Q_OBJECT

public:
    explicit MusicPlayer(QObject *parent = nullptr) : QObject(parent)
    {
        // 创建一个QMediaPlayer对象
        player = new QMediaPlayer(parent);
        // 设置播放的音频文件
        player->setMedia(QUrl::fromLocalFile("./src/xs_mtsz.mp3"));
        player->setVolume(50); //0~100音量范围, 默认是100
    }

public slots:
    // 在线程中执行的播放函数
    void play()
    {
        // 开始播放
        player->play();
    }

    // 在线程中执行的暂停函数
    void pause()
    {
        // 暂停播放
        player->pause();
    }

public:
    QMediaPlayer::State state() { return player->state(); }

private:
    QMediaPlayer *player;
};

#endif // MUSICPLAYER_H
