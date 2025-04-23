from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
import logging

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get('/summary')
def summary_api():
    try:
        video_url = request.args.get('url', '')

        if not video_url:
            return jsonify({'error': 'URL parameter is required'}), 400

        logger.info(f"Received video URL: {video_url}")

        # Extract video ID from URL
        video_id = extract_video_id(video_url)

        if not video_id:
            return jsonify({'error': 'Could not extract video ID from URL'}), 400

        logger.info(f"Extracted video ID: {video_id}")

        # Get transcript
        transcript = get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'Could not fetch transcript'}), 400

        # Generate summary
        summary = get_summary(transcript)
        return jsonify({'summary': summary}), 200

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

def extract_video_id(url):
    """
    Extracts YouTube video ID from a standard or shortened URL.
    """
    try:
        if 'v=' in url:
            video_id = url.split('v=')[1]
        else:
            video_id = url.split('/')[-1]
        return video_id.split('&')[0]
    except Exception as e:
        logger.error(f"Error extracting video ID: {str(e)}")
        return None

def get_transcript(video_id):
    """
    Fetches transcript text for a given video ID.
    """
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([entry['text'] for entry in transcript_data])
        logger.info(f"Fetched transcript ({len(transcript)} characters)")
        return transcript
    except TranscriptsDisabled:
        logger.warning("Transcripts are disabled for this video")
    except NoTranscriptFound:
        logger.warning("No transcript available for this video")
    except Exception as e:
        logger.error(f"Transcript error: {str(e)}")
    return None

def get_summary(transcript):
    """
    Generates summary using a pretrained summarization model.
    """
    try:
        summarizer = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

        summary_result = ''
        max_length = 130
        min_length = 30

        for i in range(0, len(transcript), 1000):
            chunk = transcript[i:i + 1000]
            if len(chunk) < 50:
                continue

            response = summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            summary_result += response[0]['summary_text'] + ' '

        logger.info("Summary generation complete")
        return summary_result.strip()
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return "Could not generate summary"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
