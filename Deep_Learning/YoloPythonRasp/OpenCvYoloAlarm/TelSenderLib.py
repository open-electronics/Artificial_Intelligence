from telethon import TelegramClient, events, sync

def telinit(apiid, apihash):
    client = TelegramClient('TelAlarm', apiid, apihash)
    client.connect()
    return client

def alarm(client, botaddress):
    bot = client.get_entity(botaddress)
    client.send_message(bot, '/alarm')

def sendMessage(client, mess):
    me = client.get_me()
    client.send_message(me, mess)

def sendvideo(client, filevid):
    me = client.get_me()
    client.send_file(me, filevid)

def disconnect(client):
    client.disconnect()