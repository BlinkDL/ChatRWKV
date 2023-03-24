//
// Created by diwa on 2023/3/21.
//

#pragma once
#include <QObject>
#include "ChatNetworkRequest.h"

class InteractiveMain : public QObject
{
    Q_OBJECT
public:
    explicit InteractiveMain(QObject *parent = nullptr);

    Q_INVOKABLE void sendMessage(const QString &message, bool chat);
    Q_INVOKABLE void updateTokenCount(int tokenCount);
    Q_INVOKABLE void updateTemperature(int temperature);
    Q_INVOKABLE void updateTopP(int topP);
    Q_INVOKABLE void updatePresencePenalty(int presencePenalty);
    Q_INVOKABLE void updateCountPenalty(int countPenalty);

public:
    signals:
     void sigReceivedChatMessage(const QString &text);

private:
     int m_tokenCount = 200;
     int m_temperature = 100;
     int m_topP = 80;
     int m_presencePenalty = 10;
     int m_countPenalty = 10;
     std::shared_ptr<ChatNetworkRequest> m_networkRequest;
     bool m_requestFinished = true;

};
