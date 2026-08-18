"""
================================================================
 GOZ KIRPMA TAKIP VE UYKU / DIKKAT DAGINIKLIGI TESPIT SISTEMI
 (Eye Blink Tracker & Drowsiness Detection System)
================================================================

Bu program, bilgisayar kameran araciligiyla yuzunu ve gozlerini
gercek zamanli olarak analiz eder.

Ozellikler:
  1) Gunluk goz kirpma sayaci. Istatistikler diske kaydedilir, boylece
     gunler arasi karsilastirma yapabilirsin (orn. "bugun 100 kez
     kirptim" gibi).
  2) "Calisma modu": Gozlerin kesintisiz 10 saniye kapali kalirsa, bu
     muhtemelen uykuya daldigini ya da dikkatinin ciddi sekilde
     dagildigini gosterir. Bu durumda program Opera GX tarayicisinda
     senin belirttigin YouTube videosunu acarak seni uyandirmaya
     calisir. Gozler kapali kalmaya devam ederse video belirli
     araliklarla tekrar acilir (tek seferlik degil, israrli alarm).
     Opera GX bilgisayarda bulunamazsa otomatik olarak varsayilan
     tarayicida acilir.
  3) Dusuk kirpma orani uyarisi: Ekrana uzun sure bakarken kirpma
     sikligi normalin altina duserse (dijital goz yorgunlugu / goz
     kurulugu riski gostergesidir), terminale bir uyari yazdirilir.
  4) 20-20-20 kurali hatirlaticisi: Goz sagligi icin klasik bir
     tavsiyedir - her 20 dakikada bir, 20 saniyeligine ~6 metre
     (20 fit) uzaga bakmani hatirlatir.
  5) Program acilista son 7 gunun kirpma istatistiklerini gosterir.

NOT (guncelleme): Bu surum, Google'in mediapipe kutuphanesindeki
GUNCEL "Tasks" arayuzunu (FaceLandmarker + blendshape) kullanir.
Eski "mp.solutions.face_mesh" arayuzu, mediapipe'in yakin zamanda
1.0 surumune gecmesiyle kaldirildi; bu yuzden kod o eski arayuzu
kullanan surumlerde calismaz hale gelmisti. Goz kirpma tespiti
artik elle hesaplanan bir oran yerine, modelin dogrudan urettigi
"eyeBlinkLeft" / "eyeBlinkRight" skorlarina dayaniyor (daha guvenilir
ve kalibrasyon gerektirmiyor).

Kurulum:
    pip install -r requirements.txt

Calistirma:
    python goz_takip_sistemi.py

(Ilk calistirmada, yuz analiz modeli internetten bir kere otomatik
indirilir - birkac MB, birkac saniye surer.)

Kontroller (kamera penceresi acikken):
    q -> cikis
    r -> bugunku istatistikleri sifirla

NOT (gizlilik): Tum analiz bilgisayarinda, yerel olarak yapilir.
Hicbir goruntu internete gonderilmez. Sadece kirpma SAYILARI (goruntu
degil) "goz_istatistikleri.json" dosyasina yazilir.
"""

import cv2
import mediapipe as mp
import time
import json
import os
import subprocess
import platform
import webbrowser
import urllib.request
from datetime import date

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# ============================== AYARLAR ==============================
BLINK_SCORE_THRESHOLD = 0.5       # eyeBlink skorunun (0-1) bu deger ustunde olmasi = goz kapali
CLOSED_EYES_DURATION = 10.0       # Saniye - gozler kesintisiz bu kadar kapali kalirsa tetiklenir
RETRIGGER_INTERVAL = 15.0         # Gozler kapali kalmaya devam ederse video kac saniyede bir tekrar acilsin
CONSEC_FRAMES_FOR_BLINK = 2       # Bir "kirpma" sayilmasi icin gereken ardisik kapali kare sayisi
WAKE_UP_URL = "https://www.youtube.com/watch?v=PykZLy07v9s"

STATS_FILE = "goz_istatistikleri.json"

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = "face_landmarker.task"

LOW_BLINK_RATE_PER_MIN = 10       # Dakikada bu sayidan az kirpma tespit edilirse uyari ver
LOW_BLINK_CHECK_INTERVAL = 60     # Kac saniyede bir kirpma orani kontrol edilsin
BREAK_REMINDER_INTERVAL = 20 * 60 # 20-20-20 kurali icin hatirlatma araligi (saniye)
FACE_LOST_RESET_TIME = 2.0        # Yuz bu kadar sure algilanamazsa kapali-goz sayaci sifirlanir

# Goz cevresi cizim noktalari (468 noktali yuz aginda), sadece ekrana nokta
# cizmek icin kullanilir - kirpma karari artik bunlardan hesaplanmiyor.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def ensure_model_downloaded():
    """Yuz analiz modeli (face_landmarker.task) diskte yoksa internetten indirir."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return
    print("Yuz analiz modeli indiriliyor (sadece ilk calistirmada)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model indirildi.\n")
    except Exception as e:
        print(f"[HATA] Model indirilemedi: {e}")
        print(f"Internet baglantini kontrol et, ya da su adresten elle indirip")
        print(f"scriptle ayni klasore '{MODEL_PATH}' adiyla koy:")
        print(f"  {MODEL_URL}")
        raise SystemExit(1)


def get_blink_scores(blendshapes):
    """Blendshape listesinden sol/sag goz kirpma (eyeBlink) skorlarini cikarir.
    0.0 = goz tam acik, 1.0 = goz tam kapali/kirpik."""
    left = right = 0.0
    for category in blendshapes:
        if category.category_name == "eyeBlinkLeft":
            left = category.score
        elif category.category_name == "eyeBlinkRight":
            right = category.score
    return left, right


def load_stats():
    """Kayitli istatistikleri JSON dosyasindan yukler."""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_stats(stats):
    """Istatistikleri JSON dosyasina kaydeder."""
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"[HATA] Istatistikler kaydedilemedi: {e}")


def print_recent_history(stats, days=7):
    """Son N gunun kirpma istatistiklerini terminale yazdirir."""
    if not stats:
        return
    recent_days = sorted(stats.keys(), reverse=True)[:days]
    if not recent_days:
        return
    print("Son gunlerin ozeti:")
    for d in sorted(recent_days):
        info = stats[d]
        print(f"  {d}: {info.get('blink_count', 0)} kirpma, "
              f"{info.get('wake_up_triggers', 0)} uyari tetiklendi")
    print()


def find_opera_gx_path():
    """Isletim sistemine gore Opera GX kurulum yolunu bulmaya calisir.
    Bulamazsa None doner (bu durumda varsayilan tarayici kullanilir)."""
    system = platform.system()

    if system == "Windows":
        local_appdata = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            os.path.join(local_appdata, "Programs", "Opera GX", "launcher.exe"),
            os.path.join(local_appdata, "Programs", "Opera GX", "opera.exe"),
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            "/Applications/Opera GX.app/Contents/MacOS/Opera GX",
        ]
    else:
        # Opera GX resmi olarak Linux'ta dagitilmiyor
        candidates = []

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def open_wake_up_video():
    """Calisma modu tetiklendiginde Opera GX'te (bulunamazsa varsayilan
    tarayicida) belirtilen YouTube videosunu acar."""
    opera_path = find_opera_gx_path()
    print("\a", end="")  # basit terminal biti (destekleyen terminallerde duyulur)
    try:
        if opera_path:
            subprocess.Popen([opera_path, WAKE_UP_URL])
            print(f"[UYARI] Gozler {CLOSED_EYES_DURATION:.0f} saniyedir kapali! "
                  f"Opera GX'te uyandirma videosu acildi.")
        else:
            webbrowser.open(WAKE_UP_URL)
            print(f"[UYARI] Gozler {CLOSED_EYES_DURATION:.0f} saniyedir kapali! "
                  f"Opera GX bulunamadi, varsayilan tarayicida acildi.")
    except Exception as e:
        print(f"[HATA] Tarayici acilirken sorun olustu: {e}")
        try:
            webbrowser.open(WAKE_UP_URL)
        except Exception:
            pass


def main():
    print("=" * 60)
    print(" GOZ KIRPMA TAKIP VE UYKU TESPIT SISTEMI")
    print("=" * 60)
    print("Cikmak icin    : q")
    print("Sifirlamak icin: r\n")

    ensure_model_downloaded()

    stats = load_stats()
    print_recent_history(stats)

    today = str(date.today())
    if today not in stats:
        stats[today] = {"blink_count": 0, "wake_up_triggers": 0}

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )

    try:
        landmarker = FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"[HATA] Yuz analiz modeli yuklenemedi: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HATA] Kamera acilamadi. Baska bir uygulama kamerayi kullaniyor olabilir.")
        landmarker.close()
        return

    blink_frame_counter = 0
    session_blinks = 0
    eyes_closed_since = None
    trigger_active = False
    last_trigger_time = 0.0
    no_face_since = None

    last_rate_check = time.time()
    blinks_since_rate_check = 0
    last_break_reminder = time.time()

    start_time = time.time()
    last_ts_ms = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[HATA] Kameradan goruntu alinamadi.")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int((time.time() - start_time) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            result = landmarker.detect_for_video(mp_image, ts_ms)

            face_found = bool(result.face_blendshapes)
            blink_score = 0.0

            if face_found:
                no_face_since = None
                left_score, right_score = get_blink_scores(result.face_blendshapes[0])
                blink_score = (left_score + right_score) / 2.0

                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    for idx_list in (LEFT_EYE_IDX, RIGHT_EYE_IDX):
                        for i in idx_list:
                            p = (int(lm[i].x * w), int(lm[i].y * h))
                            cv2.circle(frame, p, 2, (0, 255, 0), -1)

                eyes_closed = blink_score > BLINK_SCORE_THRESHOLD

                if eyes_closed:
                    blink_frame_counter += 1
                    if eyes_closed_since is None:
                        eyes_closed_since = time.time()

                    closed_for = time.time() - eyes_closed_since
                    should_trigger = closed_for >= CLOSED_EYES_DURATION and (
                        not trigger_active
                        or (time.time() - last_trigger_time) > RETRIGGER_INTERVAL
                    )
                    if should_trigger:
                        trigger_active = True
                        last_trigger_time = time.time()
                        stats[today]["wake_up_triggers"] += 1
                        open_wake_up_video()
                else:
                    if blink_frame_counter >= CONSEC_FRAMES_FOR_BLINK:
                        session_blinks += 1
                        blinks_since_rate_check += 1
                        stats[today]["blink_count"] += 1
                    blink_frame_counter = 0
                    eyes_closed_since = None
                    trigger_active = False
            else:
                if no_face_since is None:
                    no_face_since = time.time()
                elif time.time() - no_face_since > FACE_LOST_RESET_TIME:
                    eyes_closed_since = None
                    blink_frame_counter = 0

            # ---- ekran uzeri bilgiler ----
            if not face_found:
                status, color = "YUZ ALGILANAMADI", (0, 0, 255)
            elif blink_score > BLINK_SCORE_THRESHOLD:
                status, color = "GOZLER KAPALI", (0, 0, 255)
            else:
                status, color = "Gozler Acik", (0, 255, 0)

            cv2.putText(frame, f"Durum: {status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Kirpma skoru: {blink_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Bugunku kirpma: {stats[today]['blink_count']}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Oturum kirpma: {session_blinks}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if eyes_closed_since is not None:
                remaining = max(0.0, CLOSED_EYES_DURATION - (time.time() - eyes_closed_since))
                cv2.putText(frame, f"Uyari sayaci: {remaining:.1f}s", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("Goz Kirpma Takibi  (q: cikis, r: sifirla)", frame)

            # ---- periyodik kontroller ----
            now = time.time()
            if now - last_rate_check >= LOW_BLINK_CHECK_INTERVAL:
                rate_per_min = blinks_since_rate_check * (60.0 / LOW_BLINK_CHECK_INTERVAL)
                if rate_per_min < LOW_BLINK_RATE_PER_MIN:
                    print(f"[UYARI] Dusuk kirpma orani: {rate_per_min:.1f}/dk. "
                          f"Goz kurulugunu onlemek icin bilincli sekilde kirpistir.")
                blinks_since_rate_check = 0
                last_rate_check = now

            if now - last_break_reminder >= BREAK_REMINDER_INTERVAL:
                print("[HATIRLATMA] 20-20-20 kurali: 20 saniyeligine ~6 metre "
                      "uzaktaki bir noktaya bak.")
                last_break_reminder = now

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                stats[today] = {"blink_count": 0, "wake_up_triggers": 0}
                session_blinks = 0
                print("Bugunku istatistikler sifirlandi.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            landmarker.close()
        except Exception:
            pass
        save_stats(stats)
        print("\n" + "=" * 60)
        print(f"Oturum bitti. Bugunku toplam kirpma: {stats[today]['blink_count']}")
        print(f"Uyari tetiklenme sayisi: {stats[today]['wake_up_triggers']}")
        print(f"Istatistikler '{STATS_FILE}' dosyasina kaydedildi.")
        print("=" * 60)


if __name__ == "__main__":
    main()