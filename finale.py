#!/usr/bin/env python3
"""
DOFBOT 6-DOF Robotic Arm Controller
Controllo singolo motori e cinematica inversa per Yahboom DOFBOT con Raspberry Pi
Integrato con FastSAM per il rilevamento automatico tramite tastiera.
"""

import os
import math
import time
import cv2
import numpy as np
from typing import Tuple, List
from ultralytics import FastSAM

os.environ["DISPLAY"] = ":0"

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO non disponibile (non su Raspberry Pi)")

# =========================================================
# PARAMETRI CONFIGURAZIONE VISIVA
# =========================================================
MIN_AREA = 800

# =========================================================
# FUNZIONI DI PULIZIA E SEGMENTAZIONE (FastSAM)
# =========================================================

def filter_border_bboxes(boxes, img_shape, margin=3):
    h, w = img_shape[:2]
    out = []
    for x, y, bw, bh in boxes:
        if x <= margin or y <= margin or x + bw >= w - margin or y + bh >= h - margin:
            continue
        out.append((x, y, bw, bh))
    return out

def filter_contained_bboxes(boxes, contain_thr=0.90):
    kept = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a
        area_a = aw * ah
        contained = False
        for j, b in enumerate(boxes):
            if i == j: continue
            bx, by, bw, bh = b
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter / float(area_a + 1e-6) > contain_thr and (bw * bh) > area_a:
                contained = True
                break
        if not contained: kept.append(a)
    return kept

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter + 1e-6)

def filter_overlaps(boxes, thr=0.2):
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in boxes:
        if all(iou(b, k) <= thr for k in kept):
            kept.append(b)
    return kept

def clean_bboxes(boxes, img_shape):
    boxes = filter_border_bboxes(boxes, img_shape)
    boxes = filter_contained_bboxes(boxes)
    boxes = filter_overlaps(boxes)
    return boxes

def get_object_coordinates(img, model):
    """Analizza l'immagine con FastSAM e restituisce il centroide (x,y) del primo oggetto valido."""
    img_resized = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 70, 170)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    num, cc = cv2.connectedComponents(edges)
    psyduck_centroids = []
    for i in range(1, num):
        comp = (cc == i).astype(np.uint8) * 255
        if cv2.countNonZero(comp) < MIN_AREA: continue
        x, y, w, h = cv2.boundingRect(comp)
        if w < 25 or h < 25 or (w * h > 0.75 * 500 * 500): continue
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(comp)
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)
        ys, xs = np.where(filled > 0)
        if len(xs) > 0:
            psyduck_centroids.append((int(np.mean(xs)), int(np.mean(ys))))

    detections = []
    results = model(img_resized, iou=0.9, retina_masks=False, imgsz=512)
    masks = results[0].masks.data.cpu().numpy()

    for m in masks:
        mask = (m > 0.5).astype(np.uint8) * 255
        if cv2.countNonZero(mask) < MIN_AREA: continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            detections.append({
                "bbox": cv2.boundingRect(c),
                "contour": c,
                "center": (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            })

    if len(masks) > 0:
        areas = [m.sum() for m in masks]
        mask = (masks[np.argmax(areas)] > 0.5).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < MIN_AREA: continue
            M = cv2.moments(c)
            if M["m00"] == 0: continue
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0: continue
            detections.append({
                "bbox": (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min()),
                "contour": c,
                "center": (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            })

    boxes = [d["bbox"] for d in detections]
    filtered_boxes = clean_bboxes(boxes, img_resized.shape)

    for d in detections:
        if d["bbox"] in filtered_boxes:
            return d["center"]  # Coordinate relative allo spazio 512x512
    return None

# =========================================================
# CONVERSIONE SPAZIALE E CENTRALE CON TASTIERA
# =========================================================

def pixel_to_cm(px, py, 
                image_width=512, 
                image_height=512,
                camera_height=26.0,  
                field_width=18.6,    
                field_height=13.5):  
    """Converte i pixel rilevati nello spazio 512x512 in centimetri reali."""
    center_x = image_width / 2    # 256
    center_y = image_height / 2   # 256
    
    scale_x = field_width / image_width  
    scale_y = field_height / image_height  
    
    x_cm = (px - center_x) * scale_x
    y_cm = (py - center_y) * scale_y
    
    return (x_cm, y_cm)

def centrale(model):
    """
    Mostra il video live. All'invio della BARRA SPAZIATRICE, rileva l'oggetto 
    tramite FastSAM e ne restituisce le coordinate reali in cm.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: impossibile aprire la telecamera")
        return 0.0, 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    x_cm, y_cm = 0.0, 0.0
    print("\n[INFO] Finestra attiva. Posiziona l'oggetto e premi 'SPAZIO' per catturarlo. 'ESC' per uscire.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Errore: impossibile leggere il frame")
            break

        # Istruzioni in sovrimpressione
        cv2.putText(frame, "Premi SPAZIO per rilevare l'oggetto", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Camera DOFBOT", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Pressione TASTO SPAZIO
        if key == 32:
            print("[INPUT] Tasto Spazio premuto! Rilevamento in corso...")
            point = get_object_coordinates(frame, model)
            
            if point is not None:
                px, py = point
                # Nota: Impostiamo forzatamente 512x512 perché la funzione FastSAM lavora su quella risoluzione
                x_cm, y_cm = pixel_to_cm(px, py, image_width=512, image_height=512)
                print(f"[OK] Oggetto rilevato -> Pixel(512x512): ({px}, {py}) -> Reali: ({x_cm:.2f} cm, {y_cm:.2f} cm)")
                
                # Feedback visivo di conferma sul frame catturato
                frame_res = cv2.resize(frame, (512, 512))
                cv2.circle(frame_res, (px, py), 7, (0, 255, 0), -1)
                cv2.putText(frame_res, f"Target: {x_cm:.1f}, {y_cm:.1f} cm", (px + 10, py - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow("Camera DOFBOT", frame_res)
                cv2.waitKey(1500)  # Pausa di 1.5 secondi per mostrare il punto agganciato
                break
            else:
                print("[WARN] Nessun oggetto valido trovato nell'inquadratura. Riprova.")
        
        # Pressione TASTO ESC
        elif key == 27:
            print("[INFO] Uscita forzata dall'utente.")
            break

    # Rilascio corretto delle risorse PRIMA del return
    cap.release()
    cv2.destroyAllWindows()
    return x_cm, y_cm

# =========================================================
# CONTROLLER CINEMATICA ROBOTICA (INVARIATO)
# =========================================================

class DOFBOTController:
    def __init__(self):
        self.l1 = 0      
        self.l2 = 9.0    
        self.l3 = 9.0    
        self.l4 = 19.0   
        self.offsets = [0, 0, 0, 0, 0, 0]
        self.angle_ranges = {1: (0, 180), 2: (0, 180), 3: (0, 180), 4: (0, 180), 5: (0, 270), 6: (0, 180)}
        self.current_angles = [0, 0, 0, 0, 0, 0]
        if GPIO_AVAILABLE: self.setup_gpio()
    
    def setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        self.servo_pins = [17, 27, 22, 23, 24, 25]
        for pin in self.servo_pins: GPIO.setup(pin, GPIO.OUT)
    
    def angle_to_pulse(self, angle: float, motor_num: int) -> int:
        min_pulse, max_pulse = 500, 2500
        max_angle = self.angle_ranges[motor_num][1]
        return int(min_pulse + ((angle / max_angle) * (max_pulse - min_pulse)))
    
    def move_single_motor(self, motor_num: int, angle: float, block: bool = True):
        min_angle, max_angle = self.angle_ranges[motor_num]
        angle = max(min_angle, min(max_angle, angle)) + self.offsets[motor_num - 1]
        self.current_angles[motor_num - 1] = angle
        if GPIO_AVAILABLE:
            pulse = self.angle_to_pulse(angle, motor_num)
        else:
            print(f"Motore {motor_num}: {angle:.1f}° (simulazione)")
    
    def move_all_motors(self, angles: List[float]):
        for i, angle in enumerate(angles):
            self.move_single_motor(i + 1, angle, block=False)
    
    def cm_to_gradi(self, x: float):
        x = max(0, min(x, 6))
        punti = [(0, 180), (2, 150), (3.5, 120), (4.8, 90), (5.7, 60), (6, 0)]
        for (x1, y1), (x2, y2) in zip(punti[:-1], punti[1:]):
            if x1 <= x <= x2:
                return y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
        return 0
    
    def inverse_kinematics(self, x: float, y: float, z: float, cm: float = 6, n_it: float = 200):
        r = math.sqrt(x**2 + y**2)
        theta_1 = math.degrees(math.atan2(y, x)) + 90 if r != 0 else 0
        theta_1 = max(self.angle_ranges[1][0], min(self.angle_ranges[1][1], theta_1))
        d = math.sqrt(y**2 + x**2)
        
        def cinematica_inversa_3link(distanza, altezza, pre):
            x_target, y_target = distanza, altezza
            portata_max = self.l2 + self.l3 + self.l4
            if np.sqrt(x_target**2 + y_target**2) > portata_max:
                raise ValueError("Target oltre la portata massima")
            
            def cinematica_directa(thetas_deg):
                t1, t2, t3 = np.radians(thetas_deg)
                alpha1 = t1
                alpha2 = t1 + t2 - np.radians(90)
                alpha3 = t1 + t2 + t3 - np.radians(180)
                rx = self.l2 * np.cos(alpha1) + self.l3 * np.cos(alpha2) + self.l4 * np.cos(alpha3)
                ry = self.l2 * np.sin(alpha1) + self.l3 * np.sin(alpha2) + self.l4 * np.sin(alpha3)
                return np.array([rx, ry])
            
            def errore(thetas_deg):
                return np.sum((cinematica_directa(thetas_deg) - np.array([x_target, y_target]))**2)
            
            def gradiente(theta, eps=1e-5):
                grad = np.zeros(3)
                f0 = errore(theta)
                for i in range(3):
                    theta_p = theta.copy()
                    theta_p[i] += eps
                    grad[i] = (errore(theta_p) - f0) / eps
                return grad
            
            punti_iniziali = [[80, 70, 60], [90, 90, 90], [85, 75, 65]]
            best_theta, best_err = None, float('inf')
            
            for theta_ini in punti_iniziali:
                theta = np.array(theta_ini, dtype=float)
                lr = 0.3
                for _ in range(pre):
                    grad = gradiente(theta)
                    theta_new = np.clip(theta - lr * grad, 0, 180)
                    if errore(theta_new) >= errore(theta):
                        lr *= 0.5
                        if lr < 1e-8: break
                    else:
                        lr = min(lr * 1.02, 0.3)
                    theta = theta_new
                if errore(theta) < best_err:
                    best_err = errore(theta)
                    best_theta = theta
            return round(best_theta[0], 1), round(best_theta[1], 1), round(best_theta[2], 1)
            
        theta_2, theta_3, theta_4 = cinematica_inversa_3link(distanza=d, altezza=z, pre=n_it)
        theta_6 = self.cm_to_gradi(cm)
        return (theta_1, theta_2, theta_3, theta_4, 90, theta_6)

    def move_to_position(self, x: float, y: float, z: float, cm: float = 6, n_it: float = 200):
        angles = self.inverse_kinematics(x, y, z, cm, n_it)
        return list(angles)

    def close(self):
        if GPIO_AVAILABLE: GPIO.cleanup()


# =========================================================
# BLOCCO DI ESECUZIONE MAIN
# =========================================================

if __name__ == "__main__":
    robot = DOFBOTController()
    
    # Inizializziamo il modello FastSAM una sola volta all'avvio
    print("[INIT] Caricamento modello FastSAM...")
    sam_model = FastSAM("FastSAM-s.pt")
    
    try:
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        
        while True:
            # 1. Posizione di Home
            print("\n[ROBOT] Ritorno in posizione HOME...")
            arm.Arm_serial_servo_write6(90, 90, 0, 0, 90, 0, 5000)
            time.sleep(5)
            
            # 2. Acquisizione coordinate (Attesa della tastiera)
            x, y = centrale(sam_model)
            
            # Se l'utente chiude forzatamente con ESC (ritorna 0,0), saltiamo il ciclo di pick
            if x == 0.0 and y == 0.0:
                continue
                
            # 3. Movimento sull'oggetto rilevato
            angles = robot.move_to_position(-y + 15.3, -x+1, z=-25, cm=6) #14.5, -11
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 3000)
            time.sleep(3)
            
            # 4. Presa dell'oggetto
            apertura = 2
            ap = robot.cm_to_gradi(apertura)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], ap, 3000)
            time.sleep(3)
            
            # 5. Sollevamento
            angles = robot.move_to_position(-y + 15.3, -x+1, z=5, cm=apertura, n_it=10)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
            time.sleep(2)
            
            # 6. Rilascio nella zona di destinazione (Place)
            angles = robot.move_to_position(x=10, y=15, z=-25, cm=apertura, n_it=50)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 1300)
            time.sleep(1.3)
            
            # 7. Apertura pinza e disimpegno
            arm.Arm_serial_servo_write6(angles[0], 40, 45, 17, 90, 0, 1000)
            time.sleep(1)
            
    finally:
        robot.close()