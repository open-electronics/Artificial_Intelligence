#Android version#
import time
import subprocess
from telethon import TelegramClient, events

code = .................    #BOT hash code
apiid = .......             #Telegram API code
apihash = ...............   #Telegram API hash code

nomefile = "Alarm.mp3"

# Connection with auto reconnection forever and 60 seconds retry interval

bot = TelegramClient(None, apiid, apihash, connection_retries= -1, retry_delay=60 ).start(bot_token=code)

@bot.on(events.NewMessage(pattern='/start'))
async def start(event):
    """Send a message when the command /start is issued."""
    await event.respond('Hi! Ready to alarm.')
    raise events.StopPropagation

@bot.on(events.NewMessage(pattern='/alarm'))
async def start(event):
    """Alarm sound when the command /alarm is issued."""
    await event.respond('Alarm activated!')
 #   subprocess.call(['play-audio', nomefile])
    print("ALLARME!")
    raise events.StopPropagation

@bot.on(events.NewMessage)
async def echo(event):
    """Echo the user message."""
    text = 'Echo: <b> > '+event.text+'</b>'
    await event.respond(text, parse_mode = 'HTML')


def main():
    """Start the bot."""
    bot.run_until_disconnected()



if __name__ == '__main__':
    main()