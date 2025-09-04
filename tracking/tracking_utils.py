import cv2
import numpy as np
import pandas as pd
import subprocess
from pathlib import Path
from datetime import timedelta

"""
tracking_utils.py - Utilitas untuk tracking dan analisis video

Modul ini berisi fungsi-fungsi untuk:
- Memproses hasil tracking video
- Menganalisis data pergerakan
- Membuat visualisasi dan laporan
- Mengelola output video

Gunakan: from tracking_utils import *
"""

import matplotlib.pyplot as plt
import supervision as sv

# Initialize annotators
heat_map_annotator = sv.HeatMapAnnotator()
label_annotator = sv.LabelAnnotator()

# ================================
# FUNGSI MANAJEMEN FILE DAN VIDEO
# ================================


def get_next_sequence_id(output_path):
    """
    Menghasilkan ID urutan berikutnya untuk penamaan file

    Args:
        output_path (str): Path folder output

    Returns:
        int: ID urutan berikutnya

    Fungsi: Mencari folder dengan nama angka yang sudah ada dan memberikan
            angka berikutnya untuk folder baru
    """
    base_path = Path(output_path)
    if not base_path.exists():
        return 1

    existing_dirs = [
        d.name for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()
    ]
    if not existing_dirs:
        return 1

    existing_numbers = [int(d) for d in existing_dirs]
    return max(existing_numbers) + 1


def get_video_properties(video_path):
    """
    Mengekstrak properti video untuk analisis

    Args:
        video_path (str): Path file video

    Returns:
        dict: Dictionary berisi properti video (fps, durasi, dimensi, dll)

    Fungsi: Membaca informasi teknis video seperti frame rate, ukuran,
            dan durasi yang diperlukan untuk analisis
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Tidak dapat membuka file video: {video_path}")

    properties = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_seconds": int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        ),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }

    cap.release()
    return properties


def perform_yolo11_tracking_optimized(
    video_path, output_path, model, video_properties, timestamp
):
    """
    Melakukan tracking YOLO11 yang dioptimasi dengan BOTSORT menggunakan konfigurasi kustom

    Args:
        video_path (str): Path file video input
        output_path (str): Path folder output
        model: Model YOLO11 yang sudah dimuat
        video_properties (dict): Properti video (fps, dimensi, dll)
        timestamp (str): Timestamp untuk penamaan file

    Returns:
        tuple: (tracking_data, output_filename)
            - tracking_data (list): Data tracking per frame
            - output_filename (str): Path file video output

    Fungsi: Melakukan tracking objek person menggunakan YOLO11 dengan BOTSORT
            tracker yang sudah dikonfigurasi khusus. Menghasilkan video dengan
            anotasi dan data tracking untuk analisis lebih lanjut.

    Fitur yang digunakan:
    - ReID (Re-identification) untuk tracking yang lebih akurat
    - Global Motion Compensation untuk stabilitas tracking
    - Score Fusion untuk meningkatkan akurasi deteksi
    """
    print("🚀 Memulai tracking YOLO11 dengan konfigurasi BOTSORT kustom...")
    print("⚡ Fitur: ReID diaktifkan, Global Motion Compensation, Score Fusion")

    # Inisialisasi storage untuk data tracking
    tracking_data = []

    # Setup video writer untuk output
    output_filename = f"{output_path}{timestamp}/temp_tracked_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_filename,
        fourcc,
        video_properties["fps"],
        (video_properties["width"], video_properties["height"]),
    )

    try:
        # Melakukan tracking dengan konfigurasi kustom
        results = model.track(
            source=video_path,
            tracker="tracker.yaml",  # Menggunakan konfigurasi kustom
            show=False,
            save=False,
            classes=[0],  # Hanya kelas person
            device="cpu",  # Ganti ke 0 untuk GPU jika tersedia
            verbose=True,  # Tampilkan progress
            stream=True,  # Gunakan streaming untuk efisiensi memori
        )

        frame_count = 0
        processed_frames = 0

        # Proses hasil dengan tracking progress
        for result in results:
            try:
                # Ambil frame asli
                frame = result.orig_img.copy()

                # Konversi ke format supervision untuk pemrosesan yang konsisten
                detections = sv.Detections.from_ultralytics(result)

                # Simpan data tracking jika ada deteksi dengan ID
                if len(detections) > 0 and detections.tracker_id is not None:
                    for i, (bbox, tracker_id, confidence) in enumerate(
                        zip(
                            detections.xyxy,
                            detections.tracker_id,
                            detections.confidence,
                        )
                    ):
                        if (
                            tracker_id is not None
                        ):  # Hanya simpan objek yang berhasil di-track
                            x1, y1, x2, y2 = bbox
                            frame_w = video_properties["width"]
                            frame_h = video_properties["height"]

                            # Hitung koordinat yang dinormalisasi
                            x_center = (x1 + x2) / 2 / frame_w
                            y_center = (y1 + y2) / 2 / frame_h
                            width = (x2 - x1) / frame_w
                            height = (y2 - y1) / frame_h

                            tracking_data.append(
                                {
                                    "frame": frame_count,
                                    "tracker_id": int(tracker_id),
                                    "class": 0,  # kelas person
                                    "class_name": "person",
                                    "x_center": float(x_center),
                                    "y_center": float(y_center),
                                    "width": float(width),
                                    "height": float(height),
                                    "confidence": float(confidence),
                                    "x1": float(x1),
                                    "y1": float(y1),
                                    "x2": float(x2),
                                    "y2": float(y2),
                                }
                            )

                # Buat frame dengan anotasi
                annotated_frame = heat_map_annotator.annotate(
                    scene=frame, detections=detections
                )

                # Tambahkan label ID tracker
                if len(detections) > 0 and detections.tracker_id is not None:
                    labels = [
                        f"ID:{int(tracker_id)}" if tracker_id is not None else "ID:?"
                        for tracker_id in detections.tracker_id
                    ]

                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )

                # Tulis frame ke video
                video_writer.write(annotated_frame)

                frame_count += 1
                processed_frames += 1

                # Indikator progress
                if frame_count % 30 == 0:
                    print(
                        f"📹 Telah memproses {frame_count} frame, {len(tracking_data)} titik tracking..."
                    )

            except Exception as e:
                print(f"⚠️ Error memproses frame {frame_count}: {e}")
                continue

    except Exception as e:
        print(f"❌ Error tracking: {e}")
        return [], []

    finally:
        video_writer.release()

    print(f"✅ Tracking selesai!")
    print(f"📊 Total frame: {frame_count}")
    print(f"🎯 Record tracking: {len(tracking_data)}")

    if tracking_data:
        unique_ids = len(set([item["tracker_id"] for item in tracking_data]))
        print(f"👥 Orang unik yang di-track: {unique_ids}")
        print(
            f"📈 Rata-rata titik tracking per orang: {len(tracking_data)/unique_ids:.1f}"
        )

    return tracking_data, output_filename


def finalize_video_output(
    temp_video_file, output_path, timestamp, suffix="tracked_yolo11"
):
    """
    Mengkonversi video sementara ke format H.264 final

    Args:
        temp_video_file (str): Path file video sementara
        output_path (str): Path folder output
        timestamp (str): Timestamp untuk penamaan file
        suffix (str): Akhiran nama file

    Returns:
        str: Path file video final atau None jika gagal

    Fungsi: Menggunakan ffmpeg untuk mengkonversi video ke format yang
            lebih kompatibel dan menghapus file sementara
    """
    if not temp_video_file or not Path(temp_video_file).exists():
        print("⚠️ File video sementara tidak ditemukan!")
        return None

    # Nama file output final
    final_output = f"{output_path}{timestamp}/final_{suffix}.mp4"

    print(f"🎬 Mengkonversi ke format H.264: {final_output}")

    try:
        # Konversi menggunakan ffmpeg untuk kompatibilitas yang lebih baik
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                temp_video_file,
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-crf",
                "23",
                "-preset",
                "medium",
                "-y",
                "-loglevel",
                "error",
                final_output,
            ],
            check=True,
        )

        # Hapus file sementara
        Path(temp_video_file).unlink()

        print(f"✅ Video berhasil disimpan: {final_output}")
        return final_output

    except subprocess.CalledProcessError as e:
        print(f"❌ Error FFmpeg: {e}")
        print("📹 Menggunakan file sementara asli")
        return temp_video_file
    except Exception as e:
        print(f"❌ Error konversi: {e}")
        return temp_video_file


# ================================
# FUNGSI ANALISIS DATA TRACKING
# ================================


def generate_analysis_data(tracking_data, video_properties):
    """
    Menghasilkan analisis komprehensif dari data tracking

    Args:
        tracking_data (list): Data hasil tracking
        video_properties (dict): Properti video

    Returns:
        tuple: (DataFrame, fps, frame_width, frame_height)

    Fungsi: Mengkonversi data tracking mentah menjadi DataFrame pandas
            dengan informasi timestamp dan koordinat pixel
    """
    if not tracking_data:
        print("⚠️ Tidak ada data tracking tersedia untuk analisis")
        return None, None, None, None

    print("🔄 Memproses data tracking untuk analisis...")

    # Konversi ke DataFrame
    df = pd.DataFrame(tracking_data)

    # Hitung timestamp
    fps = video_properties["fps"]
    df["timestamp_seconds"] = df["frame"] / fps
    df["timestamp_formatted"] = df["timestamp_seconds"].apply(
        lambda x: str(timedelta(seconds=int(x)))
    )

    # Konversi ke koordinat pixel
    frame_width = video_properties["width"]
    frame_height = video_properties["height"]

    df["x_pixel"] = df["x_center"] * frame_width
    df["y_pixel"] = df["y_center"] * frame_height

    print(
        f"📊 Ditemukan {df['tracker_id'].nunique()} orang unik dalam {len(df)} deteksi"
    )

    return df, fps, frame_width, frame_height


def generate_person_log(df, fps, frame_width, frame_height):
    """
    Menghasilkan log detail untuk setiap orang

    Args:
        df (DataFrame): Data tracking yang sudah diproses
        fps (int): Frame rate video
        frame_width (int): Lebar frame video
        frame_height (int): Tinggi frame video

    Returns:
        DataFrame: Log detail untuk setiap person_id

    Fungsi: Menganalisis pergerakan, durasi, dan pola aktivitas setiap
            orang yang terdeteksi, termasuk analisis kuadran
    """
    person_log = []

    for person_id in df["tracker_id"].unique():
        person_data = df[df["tracker_id"] == person_id].sort_values("frame")

        # Statistik dasar
        first_detection = person_data.iloc[0]
        last_detection = person_data.iloc[-1]
        total_detections = len(person_data)
        avg_confidence = person_data["confidence"].mean()

        # Analisis pergerakan
        positions = person_data[["x_pixel", "y_pixel"]].values
        if len(positions) > 1:
            distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
            total_distance = np.sum(distances)
            max_distance = np.max(distances) if len(distances) > 0 else 0
        else:
            total_distance = 0
            max_distance = 0

        # Analisis kuadran
        mid_x, mid_y = frame_width / 2, frame_height / 2
        quadrant_time = {
            "top_left": 0,
            "top_right": 0,
            "bottom_left": 0,
            "bottom_right": 0,
        }

        for _, row in person_data.iterrows():
            x, y = row["x_pixel"], row["y_pixel"]
            if x < mid_x and y < mid_y:
                quadrant_time["top_left"] += 1
            elif x >= mid_x and y < mid_y:
                quadrant_time["top_right"] += 1
            elif x < mid_x and y >= mid_y:
                quadrant_time["bottom_left"] += 1
            else:
                quadrant_time["bottom_right"] += 1

        # Konversi ke detik
        for quadrant in quadrant_time:
            quadrant_time[quadrant] = quadrant_time[quadrant] / fps

        person_log.append(
            {
                "person_id": int(person_id),
                "first_detected_frame": int(first_detection["frame"]),
                "first_detected_time": first_detection["timestamp_formatted"],
                "first_detected_seconds": round(
                    first_detection["timestamp_seconds"], 2
                ),
                "last_detected_frame": int(last_detection["frame"]),
                "last_detected_time": last_detection["timestamp_formatted"],
                "last_detected_seconds": round(last_detection["timestamp_seconds"], 2),
                "total_detection_frames": total_detections,
                "detection_duration_seconds": round(
                    last_detection["timestamp_seconds"]
                    - first_detection["timestamp_seconds"],
                    2,
                ),
                "average_confidence": round(avg_confidence, 3),
                "total_movement_distance_pixels": round(total_distance, 2),
                "max_frame_movement_pixels": round(max_distance, 2),
                "time_in_top_left_seconds": round(quadrant_time["top_left"], 2),
                "time_in_top_right_seconds": round(quadrant_time["top_right"], 2),
                "time_in_bottom_left_seconds": round(quadrant_time["bottom_left"], 2),
                "time_in_bottom_right_seconds": round(quadrant_time["bottom_right"], 2),
                "avg_x_position": round(person_data["x_pixel"].mean(), 2),
                "avg_y_position": round(person_data["y_pixel"].mean(), 2),
                "x_position_std": round(person_data["x_pixel"].std(), 2),
                "y_position_std": round(person_data["y_pixel"].std(), 2),
            }
        )

    return pd.DataFrame(person_log)


def generate_heatmap_data(df, frame_width, frame_height, fps, grid_size=20):
    """
    Menghasilkan data heatmap dari hasil tracking

    Args:
        df (DataFrame): Data tracking
        frame_width (int): Lebar frame
        frame_height (int): Tinggi frame
        fps (int): Frame rate
        grid_size (int): Ukuran grid (default 20x20)

    Returns:
        DataFrame: Data heatmap per grid

    Fungsi: Membagi frame video menjadi grid dan menghitung intensitas
            aktivitas di setiap sel grid
    """
    print("🔥 Menghasilkan data heatmap...")
    heatmap_data = []

    # Buat grid
    x_bins = np.linspace(0, frame_width, grid_size + 1)
    y_bins = np.linspace(0, frame_height, grid_size + 1)

    for i in range(grid_size):
        for j in range(grid_size):
            x_min, x_max = x_bins[i], x_bins[i + 1]
            y_min, y_max = y_bins[j], y_bins[j + 1]

            # Hitung deteksi dalam sel ini
            in_cell = df[
                (df["x_pixel"] >= x_min)
                & (df["x_pixel"] < x_max)
                & (df["y_pixel"] >= y_min)
                & (df["y_pixel"] < y_max)
            ]

            detection_count = len(in_cell)
            unique_persons = (
                in_cell["tracker_id"].nunique() if detection_count > 0 else 0
            )
            avg_confidence = in_cell["confidence"].mean() if detection_count > 0 else 0

            heatmap_data.append(
                {
                    "grid_x": i,
                    "grid_y": j,
                    "x_min": round(x_min, 1),
                    "x_max": round(x_max, 1),
                    "y_min": round(y_min, 1),
                    "y_max": round(y_max, 1),
                    "x_center": round((x_min + x_max) / 2, 1),
                    "y_center": round((y_min + y_max) / 2, 1),
                    "detection_count": detection_count,
                    "unique_persons": unique_persons,
                    "avg_confidence": (
                        round(avg_confidence, 3) if detection_count > 0 else 0
                    ),
                    "time_spent_seconds": round(detection_count / fps, 2),
                }
            )

    return pd.DataFrame(heatmap_data)


def generate_summary_stats(df, person_log_df, heatmap_data_df):
    """
    Menghasilkan statistik ringkasan

    Args:
        df (DataFrame): Data tracking utama
        person_log_df (DataFrame): Log per orang
        heatmap_data_df (DataFrame): Data heatmap

    Returns:
        dict: Statistik ringkasan

    Fungsi: Menghitung berbagai metrik ringkasan seperti jumlah orang,
            durasi video, dan area paling aktif
    """
    if df is None or person_log_df.empty:
        return {}

    summary_stats = {
        "total_unique_persons": int(df["tracker_id"].nunique()),
        "total_detections": len(df),
        "video_duration_seconds": round(df["timestamp_seconds"].max(), 2),
        "average_persons_per_frame": round(
            df.groupby("frame")["tracker_id"].nunique().mean(), 2
        ),
        "max_persons_in_single_frame": int(
            df.groupby("frame")["tracker_id"].nunique().max()
        ),
        "most_active_person_id": int(
            person_log_df.loc[
                person_log_df["total_detection_frames"].idxmax(), "person_id"
            ]
        ),
        "longest_tracked_person_id": int(
            person_log_df.loc[
                person_log_df["detection_duration_seconds"].idxmax(), "person_id"
            ]
        ),
        "most_active_grid_cell": {
            "x": int(
                heatmap_data_df.loc[
                    heatmap_data_df["detection_count"].idxmax(), "grid_x"
                ]
            ),
            "y": int(
                heatmap_data_df.loc[
                    heatmap_data_df["detection_count"].idxmax(), "grid_y"
                ]
            ),
            "detections": int(heatmap_data_df["detection_count"].max()),
        },
    }

    return summary_stats


# ================================
# FUNGSI VISUALISASI
# ================================


def plot_person_timeline(person_log_df, output_path, timestamp):
    """
    Menghasilkan visualisasi timeline orang

    Args:
        person_log_df (DataFrame): Log data per orang
        output_path (str): Path output
        timestamp (str): Timestamp untuk nama file

    Fungsi: Membuat grafik timeline yang menunjukkan kapan setiap
            orang terdeteksi dalam video
    """
    if person_log_df.empty:
        print("⚠️ Tidak ada data orang untuk timeline")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(person_log_df)))

    for i, (_, person) in enumerate(person_log_df.iterrows()):
        person_id = person["person_id"]
        start_time = person["first_detected_seconds"]
        end_time = person["last_detected_seconds"]

        ax.barh(
            person_id,
            end_time - start_time,
            left=start_time,
            height=0.6,
            alpha=0.8,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Waktu (detik)", fontsize=12)
    ax.set_ylabel("ID Orang", fontsize=12)
    ax.set_title("Timeline Deteksi Orang", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        f"{output_path}{timestamp}/person_timeline_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_detection_heatmap(heatmap_data_df, output_path, timestamp, grid_size=20):
    """
    Menghasilkan visualisasi heatmap deteksi

    Args:
        heatmap_data_df (DataFrame): Data heatmap
        output_path (str): Path output
        timestamp (str): Timestamp untuk nama file
        grid_size (int): Ukuran grid

    Fungsi: Membuat heatmap yang menunjukkan area dengan aktivitas
            deteksi tertinggi dalam video
    """
    if heatmap_data_df.empty:
        print("⚠️ Tidak ada data heatmap tersedia")
        return

    heatmap_matrix = heatmap_data_df["detection_count"].values.reshape(
        grid_size, grid_size
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(heatmap_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_title(
        "Heatmap Deteksi Orang\n(Warna lebih hangat = Lebih banyak deteksi)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Posisi Grid X (Kiri → Kanan)", fontsize=12)
    ax.set_ylabel("Posisi Grid Y (Atas → Bawah)", fontsize=12)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Jumlah Deteksi", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        f"{output_path}{timestamp}/detection_heatmap_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_activity_analysis(person_log_df, output_path, timestamp):
    """
    Menghasilkan plot analisis aktivitas orang

    Args:
        person_log_df (DataFrame): Log data per orang
        output_path (str): Path output
        timestamp (str): Timestamp untuk nama file

    Fungsi: Membuat 4 subplot yang menganalisis durasi deteksi,
            jarak pergerakan, confidence score, dan jumlah deteksi
    """
    if person_log_df.empty:
        print("⚠️ Tidak ada data orang untuk analisis aktivitas")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Durasi deteksi
    ax1.hist(
        person_log_df["detection_duration_seconds"],
        bins=min(15, len(person_log_df)),
        alpha=0.7,
        edgecolor="black",
        color="skyblue",
    )
    ax1.set_xlabel("Durasi Deteksi (detik)")
    ax1.set_ylabel("Jumlah Orang")
    ax1.set_title("Distribusi Durasi Deteksi Orang")
    ax1.grid(True, alpha=0.3)

    # Jarak pergerakan
    ax2.hist(
        person_log_df["total_movement_distance_pixels"],
        bins=min(15, len(person_log_df)),
        alpha=0.7,
        edgecolor="black",
        color="lightcoral",
    )
    ax2.set_xlabel("Total Jarak Pergerakan (pixel)")
    ax2.set_ylabel("Jumlah Orang")
    ax2.set_title("Distribusi Jarak Pergerakan Orang")
    ax2.grid(True, alpha=0.3)

    # Distribusi confidence
    ax3.hist(
        person_log_df["average_confidence"],
        bins=min(15, len(person_log_df)),
        alpha=0.7,
        edgecolor="black",
        color="lightgreen",
    )
    ax3.set_xlabel("Skor Confidence Rata-rata")
    ax3.set_ylabel("Jumlah Orang")
    ax3.set_title("Distribusi Skor Confidence Rata-rata")
    ax3.grid(True, alpha=0.3)

    # Jumlah deteksi per orang
    ax4.bar(
        person_log_df["person_id"],
        person_log_df["total_detection_frames"],
        alpha=0.7,
        edgecolor="black",
        color="orange",
    )
    ax4.set_xlabel("ID Orang")
    ax4.set_ylabel("Total Frame Deteksi")
    ax4.set_title("Jumlah Deteksi per Orang")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_path}{timestamp}/activity_analysis_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def display_summary_stats(summary_stats):
    """
    Menampilkan statistik ringkasan

    Args:
        summary_stats (dict): Dictionary statistik ringkasan

    Fungsi: Menampilkan ringkasan statistik dalam format yang
            mudah dibaca dengan emoji dan formatting
    """
    if not summary_stats:
        print("⚠️ Tidak ada statistik ringkasan tersedia")
        return

    print("\n" + "=" * 50)
    print("📊 STATISTIK RINGKASAN DETEKSI")
    print("=" * 50)
    print(f"👥 Total orang unik terdeteksi: {summary_stats['total_unique_persons']}")
    print(f"🎯 Total deteksi: {summary_stats['total_detections']:,}")
    print(f"⏱  Durasi video: {summary_stats['video_duration_seconds']:.1f} detik")
    print(
        f"📈 Rata-rata orang per frame: {summary_stats['average_persons_per_frame']:.2f}"
    )
    print(
        f"🏆 Maksimum orang dalam satu frame: {summary_stats['max_persons_in_single_frame']}"
    )
    print(f"🔥 Orang paling aktif: ID #{summary_stats['most_active_person_id']}")
    print(f"⏰ Orang terlama dilacak: ID #{summary_stats['longest_tracked_person_id']}")
    print(
        f"📍 Titik terpanas: Grid ({summary_stats['most_active_grid_cell']['x']}, {summary_stats['most_active_grid_cell']['y']}) dengan {summary_stats['most_active_grid_cell']['detections']} deteksi"
    )
    print("=" * 50)


# ================================
# INFORMASI MODUL
# ================================


def show_functions_info():
    """
    Menampilkan informasi tentang semua fungsi dalam modul
    """
    print("📚 DAFTAR FUNGSI TRACKING_UTILS")
    print("=" * 60)
    print("🗂️  MANAJEMEN FILE & VIDEO:")
    print("   • get_next_sequence_id() - ID urutan untuk folder")
    print("   • get_video_properties() - Ekstrak properti video")
    print("   • finalize_video_output() - Konversi video ke H.264")
    print()
    print("📊 ANALISIS DATA:")
    print("   • generate_analysis_data() - Proses data tracking")
    print("   • generate_person_log() - Log detail per orang")
    print("   • generate_heatmap_data() - Data heatmap grid")
    print("   • generate_summary_stats() - Statistik ringkasan")
    print()
    print("📈 VISUALISASI:")
    print("   • plot_person_timeline() - Timeline deteksi")
    print("   • plot_detection_heatmap() - Heatmap aktivitas")
    print("   • plot_activity_analysis() - Analisis 4 subplot")
    print("   • display_summary_stats() - Tampilkan ringkasan")
    print()
    print("❓ BANTUAN:")
    print("   • show_functions_info() - Tampilkan info ini")
    print("=" * 60)


if __name__ == "__main__":
    show_functions_info()
