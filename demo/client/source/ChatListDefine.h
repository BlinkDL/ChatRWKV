//
// Created by diwa on 2023/3/21.
//
#pragma once
#include <QObject>
#include <QSharedPointer>

class ChatData : public QObject
{
    Q_OBJECT
public:
    using QObject::QObject;
    enum DataType
    {
        Text
    };
    Q_ENUM(DataType)

    enum DataStatus
    {
        DataError,
        DataReady,
        ParseOn,
        ParseSuccess,
        ParseError
    };
    Q_ENUM(DataStatus)
};

struct ChatDataBasic
{
    qint64 id;
    QString user;
    QString sender;
    qint64 datetime;
    ChatData::DataType type = ChatData::Text;
    ChatData::DataStatus status = ChatData::DataError;
    virtual ~ChatDataBasic(){}
};

struct ChatDataText : public ChatDataBasic
{
    QString text;
};


