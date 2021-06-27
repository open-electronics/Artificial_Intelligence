import GetConfigLib as cfg
from telethon import TelegramClient, events, sync

sessionf = 'TelAlarm'

if __name__ == "__main__":
    config = cfg.readConfigParam()
    telid = cfg.getValueInt(config, 'telegramid')
    telhash = cfg.getValue(config, 'telegramash')
    with TelegramClient(sessionf, telid, telhash) as client:
        client.loop.run_until_complete(client.send_message('me', 'Session '+sessionf+' created!'))
    print('Session '+sessionf+' created!')
