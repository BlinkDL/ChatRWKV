// ChatNetworkRequest.h

#pragma  once
#include <QObject>
#include <QNetworkAccessManager>
#include <QThreadPool>
#include <QUrlQuery>

class ChatNetworkRequest : public QObject
{
Q_OBJECT
public:
    explicit ChatNetworkRequest(QObject *parent = nullptr);

    void sendGetRequest(const QUrl &url, const QUrlQuery &params = QUrlQuery());

signals:
    void responseReceived(const QByteArray &data, bool success);

private:
    QNetworkAccessManager m_networkAccessManager;
    QThreadPool m_threadPool;
};

