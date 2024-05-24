from groq import Groq
import logging

def summarize_video(api_key, frame_descriptions, model):
    try:
        client = Groq(api_key=api_key)
        messages = [
            {
                "role": "user",
                "content": f"Here are the descriptions of some frames from the video: {frame_descriptions}. Summarize this content such that it explains what the video is about. Return only the summary and do not reply anything else."
            }
        ]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.1,
            max_tokens=2000,
            top_p=1,
            stop=None,
            stream=False,
        )
        return chat_completion.choices[0].message.content

    except Exception as e:
        logging.error(f"Error in summarizing video: {e}")
        return "Error in generating summary."
