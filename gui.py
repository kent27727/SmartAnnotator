"""
Dark Circle Auto-Annotation Tool - Web GUI
==========================================
Gradio tabanlı kullanıcı arayüzü
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import shutil

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    print("❌ Gradio yüklü değil!")
    print("Yüklemek için: pip install gradio")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics yüklü değil!")
    sys.exit(1)

import config
from utils import (
    load_yolo_segmentation_label,
    save_yolo_segmentation_label,
    mask_to_polygon,
    visualize_annotation
)


# Global model
model = None


def load_model():
    """Model yükle"""
    global model
    model_path = config.MODELS_DIR / "latest_model.pt"
    
    if model_path.exists():
        model = YOLO(str(model_path))
        return f"✅ Model yüklendi: {model_path.name}"
    else:
        return "❌ Model bulunamadı! Önce model eğitin."


def predict_single(image, confidence: float) -> Tuple[np.ndarray, str]:
    """
    Tek görsel için tahmin yap
    
    Args:
        image: Input görsel (numpy array)
        confidence: Confidence threshold
        
    Returns:
        Annotated image, status message
    """
    global model
    
    if model is None:
        return image, "❌ Model yüklenmedi!"
    
    if image is None:
        return None, "❌ Görsel yükleyin!"
    
    # Tahmin yap
    results = model.predict(
        source=image,
        conf=confidence,
        retina_masks=True,
        verbose=False
    )
    
    if not results or len(results) == 0:
        return image, "❌ Tahmin yapılamadı!"
    
    result = results[0]
    
    # Sonucu görselleştir
    annotated = result.plot()
    
    # İstatistikler
    n_detections = 0
    avg_conf = 0
    
    if result.boxes is not None:
        n_detections = len(result.boxes)
        if n_detections > 0:
            avg_conf = float(result.boxes.conf.mean())
    
    status = f"✅ {n_detections} tespit bulundu (Ortalama güven: {avg_conf:.2f})"
    
    return annotated, status


def batch_annotate(confidence: float, max_images: int, progress=gr.Progress()) -> str:
    """
    Toplu otomatik etiketleme
    
    Args:
        confidence: Confidence threshold
        max_images: Maksimum görsel sayısı
        
    Returns:
        Status message
    """
    global model
    
    if model is None:
        return "❌ Model yüklenmedi!"
    
    from auto_annotate import AutoAnnotator
    
    try:
        annotator = AutoAnnotator()
        
        stats = annotator.annotate_batch(
            images_dir=config.RAW_IMAGES_DIR,
            confidence_threshold=confidence,
            max_images=int(max_images) if max_images > 0 else None,
            save_visualizations=True
        )
        
        return f"""
✅ Otomatik etiketleme tamamlandı!

📊 Sonuçlar:
- Toplam işlenen: {stats['total_processed']}
- Yüksek güvenilirlik: {stats['high_confidence']}
- Düşük güvenilirlik: {stats['low_confidence']}
- Tespit yok: {stats['no_detection']}
- Hatalar: {stats['errors']}

📁 Çıktı: {config.AUTO_ANNOTATIONS_DIR}
        """
    except Exception as e:
        return f"❌ Hata: {str(e)}"


def get_dataset_stats() -> str:
    """Dataset istatistiklerini al"""
    
    stats = []
    
    # Raw images
    raw_count = 0
    if config.RAW_IMAGES_DIR.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            raw_count += len(list(config.RAW_IMAGES_DIR.glob(f"*{ext}")))
    stats.append(f"📁 Ham Görseller: {raw_count}")
    
    # Manual annotations
    manual_count = 0
    if config.MANUAL_ANNOTATIONS_DIR.exists():
        manual_count = len(list(config.MANUAL_ANNOTATIONS_DIR.rglob("*.txt")))
    stats.append(f"📝 Manuel Annotation: {manual_count}")
    
    # Auto annotations
    auto_count = 0
    auto_labels = config.AUTO_ANNOTATIONS_DIR / "labels"
    if auto_labels.exists():
        auto_count = len(list(auto_labels.glob("*.txt")))
    stats.append(f"🤖 Otomatik Annotation: {auto_count}")
    
    # Model status
    model_exists = (config.MODELS_DIR / "latest_model.pt").exists()
    stats.append(f"🧠 Model: {'✅ Mevcut' if model_exists else '❌ Yok'}")
    
    # Final dataset
    final_train = config.FINAL_DATASET_DIR / "train" / "images"
    final_count = 0
    if final_train.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            final_count += len(list(final_train.glob(f"*{ext}")))
    stats.append(f"📦 Final Dataset: {final_count}")
    
    return "\n".join(stats)


def get_sample_images() -> List[str]:
    """Örnek görselleri al"""
    samples = []
    
    if config.RAW_IMAGES_DIR.exists():
        for ext in ['.jpg', '.jpeg', '.png']:
            for img in config.RAW_IMAGES_DIR.glob(f"*{ext}"):
                samples.append(str(img))
                if len(samples) >= 20:
                    break
            if len(samples) >= 20:
                break
    
    return samples


def create_gui():
    """Gradio GUI oluştur"""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="Dark Circle Auto-Annotation Tool") as demo:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎯 Dark Circle Auto-Annotation Tool</h1>
            <p>YOLOv11 Segmentation için Otomatik Etiketleme Sistemi</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Tek Görsel Tahmin
            with gr.Tab("🔍 Tek Görsel Tahmin"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="Görsel Yükle", type="numpy")
                        confidence_slider = gr.Slider(
                            minimum=0.1, 
                            maximum=0.95, 
                            value=0.5, 
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        predict_btn = gr.Button("🚀 Tahmin Yap", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Sonuç")
                        status_text = gr.Textbox(label="Durum", interactive=False)
                
                predict_btn.click(
                    fn=predict_single,
                    inputs=[input_image, confidence_slider],
                    outputs=[output_image, status_text]
                )
            
            # Tab 2: Toplu Etiketleme
            with gr.Tab("🤖 Toplu Etiketleme"):
                gr.Markdown("""
                ### Otomatik Etiketleme
                Eğitilmiş modeli kullanarak tüm görselleri otomatik etiketleyin.
                """)
                
                with gr.Row():
                    batch_confidence = gr.Slider(
                        minimum=0.1, 
                        maximum=0.95, 
                        value=0.5, 
                        step=0.05,
                        label="Confidence Threshold"
                    )
                    max_images_slider = gr.Slider(
                        minimum=0, 
                        maximum=1000, 
                        value=0, 
                        step=10,
                        label="Maksimum Görsel (0 = tümü)"
                    )
                
                batch_btn = gr.Button("🚀 Toplu Etiketleme Başlat", variant="primary")
                batch_result = gr.Textbox(label="Sonuç", lines=10, interactive=False)
                
                batch_btn.click(
                    fn=batch_annotate,
                    inputs=[batch_confidence, max_images_slider],
                    outputs=[batch_result]
                )
            
            # Tab 3: İstatistikler
            with gr.Tab("📊 İstatistikler"):
                stats_display = gr.Textbox(
                    label="Dataset İstatistikleri", 
                    lines=8, 
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 Yenile")
                
                refresh_btn.click(
                    fn=get_dataset_stats,
                    outputs=[stats_display]
                )
                
                # Sayfa yüklendiğinde istatistikleri göster
                demo.load(
                    fn=get_dataset_stats,
                    outputs=[stats_display]
                )
            
            # Tab 4: Ayarlar
            with gr.Tab("⚙️ Ayarlar"):
                gr.Markdown("""
                ### Konfigürasyon
                """)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"""
                        **Dizinler:**
                        - Ham Görseller: `{config.RAW_IMAGES_DIR}`
                        - Manuel Annotation: `{config.MANUAL_ANNOTATIONS_DIR}`
                        - Otomatik Annotation: `{config.AUTO_ANNOTATIONS_DIR}`
                        - Modeller: `{config.MODELS_DIR}`
                        - Final Dataset: `{config.FINAL_DATASET_DIR}`
                        """)
                    
                    with gr.Column():
                        gr.Markdown(f"""
                        **Model Ayarları:**
                        - YOLO Model: `{config.YOLO_SEG_MODEL}`
                        - Image Size: `{config.IMAGE_SIZE}`
                        - Batch Size: `{config.BATCH_SIZE}`
                        - Epochs: `{config.INITIAL_TRAINING_EPOCHS}`
                        """)
                
                model_status = gr.Textbox(label="Model Durumu", interactive=False)
                load_model_btn = gr.Button("📥 Model Yükle")
                
                load_model_btn.click(
                    fn=load_model,
                    outputs=[model_status]
                )
        
        # Footer
        gr.Markdown("""
        ---
        💡 **Kullanım Adımları:**
        1. Roboflow'da 200-300 görsel etiketleyin
        2. Terminal'de `python main.py` ile model eğitin
        3. Bu arayüzden otomatik etiketleme yapın
        4. Final dataset'i export edin
        """)
    
    return demo


def main():
    """GUI'yi başlat"""
    
    print("\n" + "="*50)
    print("🎯 DARK CIRCLE AUTO-ANNOTATION GUI")
    print("="*50)
    
    # Dizinleri oluştur
    config.create_directories()
    
    # Model yükle
    load_model()
    
    # GUI oluştur ve başlat
    demo = create_gui()
    
    print("\n🌐 Web arayüzü başlatılıyor...")
    print("   Tarayıcınızda otomatik açılacak")
    print("   Manuel erişim: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # True yaparsanız public link oluşturur
        inbrowser=True
    )


if __name__ == "__main__":
    main()

