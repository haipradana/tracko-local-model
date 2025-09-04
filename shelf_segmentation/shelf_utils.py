import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk memecah video menjadi frame-frame
def extract_frames_from_video(video_path, output_dir):
    """
    Ekstraksi frame dari video
    """
    # Buka video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video_path}')
    
    # Dapatkan informasi video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f'Video info: {total_frames} frames, {fps} FPS, {width}x{height}')
    
    frame_count = 0
    extracted_frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simpan frame
        frame_filename = f'frame_{frame_count:06d}.jpg'
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        extracted_frames.append(frame_path)
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f'Extracted {frame_count}/{total_frames} frames')
    
    cap.release()
    print(f'Total {frame_count} frames extracted')
    
    return extracted_frames, fps, width, height

# Fungsi untuk menjalankan segmentasi pada setiap frame
def process_frames_with_model(frame_paths, model, output_dir):
    """
    Proses setiap frame dengan model YOLO
    """
    processed_frames = []
    total_frames = len(frame_paths)
    
    print(f'Processing {total_frames} frames with YOLO model...')
    
    for i, frame_path in enumerate(frame_paths):
        # Load frame
        frame = cv2.imread(frame_path)
        
        # Jalankan prediksi
        results = model(frame)
        
        # Gambar hasil deteksi pada frame
        annotated_frame = results[0].plot()
        
        # Simpan frame yang sudah dianotasi
        output_filename = f'result_{i:06d}.jpg'
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, annotated_frame)
        processed_frames.append(output_path)
        
        # Progress update
        if (i + 1) % 50 == 0 or (i + 1) == total_frames:
            print(f'Processed {i + 1}/{total_frames} frames')
    
    return processed_frames

# Fungsi untuk menggabungkan frame kembali menjadi video
def create_video_from_frames(frame_paths, output_video_path, fps, width, height):
    """
    Gabungkan frame-frame menjadi video
    """
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    print(f'Creating video: {output_video_path}')
    print(f'Video specs: {fps} FPS, {width}x{height}')
    
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        
        # Resize frame jika diperlukan
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        
        out.write(frame)
        
        if (i + 1) % 100 == 0:
            print(f'Written {i + 1}/{len(frame_paths)} frames to video')
    
    out.release()
    print(f'Video created successfully: {output_video_path}')

# Tampilkan beberapa hasil frame untuk preview
def show_sample_results(original_frames, processed_frames, num_samples=3):
    """
    Tampilkan perbandingan frame asli vs hasil segmentasi
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    # Pilih frame secara acak untuk ditampilkan
    sample_indices = np.linspace(0, len(original_frames)-1, num_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        # Frame asli
        original_img = cv2.imread(original_frames[idx])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f'Original Frame {idx}')
        axes[0, i].axis('off')
        
        # Frame hasil segmentasi
        processed_img = cv2.imread(processed_frames[idx])
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        axes[1, i].imshow(processed_img)
        axes[1, i].set_title(f'Segmented Frame {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()