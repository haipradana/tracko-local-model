import torch
import cv2
import numpy as np
import csv
from collections import defaultdict
from decord import VideoReader, cpu
import os

def get_next_run_number(base_path="hasil"):
    """
    Mencari nomor run berikutnya berdasarkan folder yang sudah ada
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)
        return 1
    
    # Cari semua folder dengan format angka 3 digit
    existing_runs = []
    for item in os.listdir(base_path):
        folder_path = os.path.join(base_path, item)
        if os.path.isdir(folder_path) and item.isdigit() and len(item) == 3:
            existing_runs.append(int(item))
    
    # Return nomor run berikutnya
    return max(existing_runs, default=0) + 1

# --- Fungsi Membersihkan Prediksi ---
def merge_consecutive_predictions(predictions, min_duration_frames=0):
    """
    Menggabungkan prediksi aksi yang berurutan dan identik menjadi satu event.
    Juga memfilter event yang durasinya terlalu pendek.
    """
    if not predictions:
        return []

    merged = []
    current_event = predictions[0].copy()

    for next_pred in predictions[1:]:
        if next_pred['pred'] == current_event['pred']:
            current_event['end'] = next_pred['end']
        else:
            merged.append(current_event)
            current_event = next_pred.copy()

    merged.append(current_event)

    final_predictions = [
        event for event in merged
        if (event['end'] - event['start']) >= min_duration_frames
    ]

    return final_predictions

def predict_multiperson_video(video_path, output_path, yolo, action_classifier, processor, device, clip_len=16, output_csv_path=None):
    id2label = action_classifier.config.id2label
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = vr.get_avg_fps()

    # 1. Lacak semua orang dalam video menggunakan YOLO
    print("Memulai pelacakan orang...")
    tracking_results = yolo.track(source=video_path, persist=True, tracker="bytetrack.yaml", classes=[0], stream=True)
    person_tracks = defaultdict(list)
    for frame_idx, result in enumerate(tracking_results):
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, track_ids):
                person_tracks[track_id].append({'frame': frame_idx, 'bbox': box})
    print(f"{len(person_tracks)} orang berhasil dilacak.")

    # 2: Klasifikasi aksi mentah untuk setiap jejak orang
    print("Memulai klasifikasi aksi mentah...")
    raw_action_predictions = defaultdict(list)
    for track_id, data in person_tracks.items():
        if len(data) < clip_len: continue
        for i in range(0, len(data) - clip_len + 1, clip_len // 2):
            clip_data = data[i : i + clip_len]
            frame_indices = [d['frame'] for d in clip_data]
            clip_frames = vr.get_batch(frame_indices).asnumpy()

            cropped_frames = [frame[int(d['bbox'][1]):int(d['bbox'][3]), int(d['bbox'][0]):int(d['bbox'][2])] for frame, d in zip(clip_frames, clip_data) if d['bbox'][3] > d['bbox'][1] and d['bbox'][2] > d['bbox'][0]]
            if not cropped_frames: continue

            inputs = processor(cropped_frames, return_tensors="pt")
            with torch.no_grad():
                outputs = action_classifier(pixel_values=inputs.pixel_values.to(device))

            pred_id = outputs.logits.argmax(-1).item()
            raw_action_predictions[track_id].append({'start': clip_data[0]['frame'], 'end': clip_data[-1]['frame'], 'pred': pred_id})
    print("Klasifikasi mentah selesai.")

    # 3: Bersihkan dan filter prediksi
    print("Membersihkan dan menggabungkan prediksi aksi...")
    action_predictions = {}
    min_duration_seconds = 0.4
    min_duration_frames = int(min_duration_seconds * fps)
    for track_id, preds in raw_action_predictions.items():
        action_predictions[track_id] = merge_consecutive_predictions(preds, min_duration_frames)

    # 4: Hitung dan simpan rekap CSV dari data yang sudah bersih
    if output_csv_path:
        print(f"Menyimpan rekap aksi ke {output_csv_path}...")
        action_counts = defaultdict(int)
        for track_id, predictions in action_predictions.items():
            for pred_info in predictions:
                action_counts[id2label[pred_info['pred']]] += 1

        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Action', 'Count'])
                for action, count in sorted(action_counts.items()):
                    writer.writerow([action, count])
            print("Rekap aksi berhasil disimpan.")
        except Exception as e:
            print(f"Gagal menyimpan file CSV: {e}")

    # 5: Visualisasikan hasil ke video baru
    print("Membuat video output dengan anotasi...")
    height, width, _ = vr[0].shape
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_idx in range(len(vr)):
        frame_bgr = cv2.cvtColor(vr[frame_idx].asnumpy(), cv2.COLOR_RGB2BGR)
        for track_id, data in person_tracks.items():
            for track_point in data:
                if track_point['frame'] == frame_idx:
                    x1, y1, x2, y2 = map(int, track_point['bbox'])
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    current_action = "..."
                    if track_id in action_predictions:
                        for pred in action_predictions[track_id]:
                            if pred['start'] <= frame_idx <= pred['end']:
                                current_action = id2label[pred['pred']].replace("_", " ")
                                break
                    label_text = f"ID: {track_id} | {current_action}"
                    cv2.putText(frame_bgr, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
        video_writer.write(frame_bgr)
    video_writer.release()
    print("Video output berhasil dibuat!")