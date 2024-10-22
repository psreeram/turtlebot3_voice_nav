import aiohttp
import openai
import os

# TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"
# OPENAI_API_KEY = "OPENAI_API_KEY"

openai.api_key = os.getenv('OPENAI_API_KEY')

async def transcribe_voice(voice_data):
    try:
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('file', voice_data, filename='voice.ogg', content_type='audio/ogg')
            form.add_field('model', 'whisper-1')
            
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            
            async with session.post('https://api.openai.com/v1/audio/transcriptions', data=form, headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['text']
                else:
                    return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None