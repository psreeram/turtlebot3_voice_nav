import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from voice_transcribe import transcribe_voice, TELEGRAM_BOT_TOKEN
from react_agent import run_react_agent

WELCOME_MESSAGE = (
    "Welcome to the TurtleBot3 Voice Navigation Bot! 🤖\n\n"
    "You can control the robot by:\n"
    "1. Sending a voice message with your navigation command\n"
    "2. Typing your command as a text message\n\n"
    "For example, try saying or typing 'Go to the kitchen' or 'Navigate to the living room'."
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! 👋\n\n{WELCOME_MESSAGE}"
    )

async def handle_voice(update: Update, context):
    if not context.user_data.get('welcomed'):
        await update.message.reply_text(WELCOME_MESSAGE)
        context.user_data['welcomed'] = True

    voice_file = await update.message.voice.get_file()
    voice_data = await voice_file.download_as_bytearray()
    
    transcription = await transcribe_voice(voice_data)
    
    if transcription:
        await update.message.reply_text(f"I heard: {transcription}")
        try:
            response = await run_react_agent(transcription)
            if response and response.strip():  # Check if response is non-empty
                # Split response into smaller chunks if needed
                chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                for chunk in chunks:
                    if chunk.strip():  # Only send non-empty chunks
                        await update.message.reply_text(chunk)
            else:
                await update.message.reply_text("I couldn't generate a proper response. Please try again.")
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            await update.message.reply_text("Sorry, there was an error processing your request. Please try again.")
    else:
        await update.message.reply_text("Sorry, there was an error understanding your message. Please try again.")

async def handle_text(update: Update, context):
    if not context.user_data.get('welcomed'):
        await update.message.reply_text(WELCOME_MESSAGE)
        context.user_data['welcomed'] = True

    user_input = update.message.text
    try:
        response = await run_react_agent(user_input)
        if response and response.strip():  # Check if response is non-empty
            # Split response into smaller chunks if needed
            chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
            for chunk in chunks:
                if chunk.strip():  # Only send non-empty chunks
                    await update.message.reply_text(chunk)
        else:
            await update.message.reply_text("I couldn't generate a proper response. Please try again.")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        await update.message.reply_text("Sorry, there was an error processing your request. Please try again.")

def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    
    print("TurtleBot3 Voice Navigation Bot is now online!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()