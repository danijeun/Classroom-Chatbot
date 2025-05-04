import gradio as gr
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import subprocess
import os
import ollama
import pandas as pd
from typing import List, Dict, Generator, Tuple, Optional

# Initialize with correct model
encoder = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings
client = QdrantClient("http://localhost:6333")

# Load CSV without headers and specify column names
df = pd.read_csv("processed_sentences.csv", header=None, 
                 names=["text", "start_time", "end_time", "video_name"])

# Now you can access the columns by name
df['text'] = df['text'].str.strip()

# Then use the correct column name
df['text'] = df['text'].str.strip()

def setup_collection():
    """Initialize collection with correct parameters"""
    try:
        collection_info = client.get_collection("video_sentences")
        print(f"Collection exists: {collection_info}")
    except Exception as e:
        print("Creating collection 'video_sentences'")
        client.create_collection(
            collection_name="video_sentences",
            vectors_config=models.VectorParams(
                size=encoder.get_sentence_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        )
        
        # Batch insert points
        batch_size = 100
        points = []
        for idx, row in df.iterrows():
            points.append(models.PointStruct(
                id=idx,
                vector=encoder.encode(row['text']).tolist(),
                payload={
                    "video": row['video'],
                    "start": row['start'],
                    "end": row['end'],
                    "text": row['text']
                }
            ))
            if len(points) >= batch_size:
                client.upsert("video_sentences", points=points)
                points = []
        
        if points:
            client.upsert("video_sentences", points=points)
        
        print(f"Collection created with {len(df)} segments")

setup_collection()

def search_videos(query: str, top_k: int = 5) -> List[Dict]:
    """Search using current Qdrant API"""
    encoded_query = encoder.encode(query).tolist()
    
    try:
        results = client.search(
            collection_name="video_sentences",
            query_vector=encoded_query,  # Correct API usage
            limit=top_k,
            with_payload=True
        )
        
        return sorted([
            {
                "video": hit.payload["video"],
                "start": hit.payload["start"],
                "end": hit.payload["end"],
                "text": hit.payload["text"],
                "score": hit.score
            }
            for hit in results
        ], key=lambda x: x['score'], reverse=True)
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

def find_video_path(video_id: str) -> Optional[str]:
    """Find video file with multiple path options"""
    possible_paths = [
        f"videos/videos/{video_id}/video.mp4",
        f"videos/{video_id}/video.mp4"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def create_clip(video_path: str, start: str, end: str, output_dir: str = "clips") -> Optional[str]:
    """Generate video clip with robust error handling"""
    os.makedirs(output_dir, exist_ok=True)
    safe_start = start.replace(":", "-").replace(".", "-")
    safe_end = end.replace(":", "-").replace(".", "-")
    output_path = os.path.join(output_dir, f"clip_{safe_start}_{safe_end}.mp4")
    
    try:
        subprocess.run([
            "ffmpeg",
            "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-ss", start,
            "-to", end,
            "-c:v", "copy",
            "-c:a", "copy",
            output_path
        ], check=True, capture_output=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Clip creation failed: {e.stderr.decode()}")
        return None

def generate_response(query: str) -> Generator[Tuple[str, str], None, None]:
    """Generate comprehensive response"""
    results = search_videos(query, top_k=5)
    
    if not results:
        yield "No relevant video segments found.", None
        return
    
    context = "Relevant Video Segments:\n" + "\n".join(
        f"- {res['text']} (Timestamp: {res['start']}-{res['end']})"
        for res in results
    )
    
    prompt = (
        "Using these video segments:\n"
        f"{context}\n\n"
        f"Question: {query}\n"
        "Provide a detailed answer with timestamp references.\n"
        "Answer:"
    )
    
    full_response = ""
    try:
        for chunk in ollama.generate(
            model='llama3',
            prompt=prompt,
            stream=True,
            options={'temperature': 0.3}
        ):
            full_response += chunk['response']
            yield full_response, None
    except Exception as e:
        print(f"LLM error: {str(e)}")
        full_response = "Error generating response. Please try again."
        yield full_response, None
    
    best_result = results[0]
    video_path = find_video_path(best_result["video"])
    if video_path:
        clip_path = create_clip(video_path, best_result["start"], best_result["end"])
        yield full_response, clip_path if clip_path else None
    else:
        yield full_response, None

# Gradio Interface
with gr.Blocks(title="Video Knowledge System") as app:
    gr.Markdown("## Video Knowledge System")
    
    with gr.Row():
        query_input = gr.Textbox(label="Your Question", placeholder="Ask about video content...")
        submit_btn = gr.Button("Search", variant="primary")
    
    with gr.Row():
        response_output = gr.Textbox(label="Answer", interactive=False, lines=8)
        video_output = gr.Video(label="Relevant Clip")
    
    gr.Examples(
        examples=[
            ["Explain how the brain processes images"],
            ["Describe image representation in computer vision"],
            ["Explain the relationship between images and matrices"]
        ],
        inputs=query_input
    )
    
    submit_btn.click(
        generate_response,
        inputs=query_input,
        outputs=[response_output, video_output]
    )

if __name__ == "__main__":
    app.launch(share=True)