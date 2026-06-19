#!/usr/bin/env python3
"""
DOFBOT 6-DOF Robotic Arm Controller
Controllo singolo motori e cinematica inversa per Yahboom DOFBOT con Raspberry Pi

Configurazione motori:
- Motore 1: Base, rotazione 0-180°
- Motore 2-4: Uno sopra l'altro, 0-180° ciascuno
- Motore 5: Sopra motore 4, 0-270°
- Motore 6: Pinza, 0-180° (mantenere a 180° per afferrare)

Distanze reali (da te misurate):
- Motore 2 → Motore 3: 9 centimetri
- Motore 3 → Motore 4: 9 centimetri
- Motore 5 → Motore 6: 19 centimetri
"""

import random
import math
import cv2

import os
os.environ["DISPLAY"] = ":0"

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO non disponibile (non su Raspberry Pi)")

import numpy as np
from typing import Tuple, List


    # Funzione di conversione pixel -> cm
def pixel_to_cm(px, py, 
                image_width=640, 
                image_height=480,
                camera_height=13.0,  # cm
                field_width=10.0,    # cm (larghezza campo visivo reale)
                field_height=7.5):  # cm (altezza campo visivo reale)
    center_x = image_width / 2  # 320
    center_y = image_height / 2  # 240
    
    scale_x = field_width / image_width  # cm/px
    scale_y = field_height / image_height  # cm/px
    
    x_cm = (px - center_x) * scale_x
    y_cm = (py - center_y) * scale_y
    
    return (x_cm, y_cm)


# Variabile globale per salvare le coordinate del click
click_coordinates = None

def onMouse(event, x, y, flags, param):
    """Callback per catturare il click mouse"""
    if event == cv2.EVENT_LBUTTONDOWN:
        global click_coordinates
        click_coordinates = (x, y)
        print(f"Cliccato in pixel: (x={x}, y={y})")


def centrale():
    global click_coordinates
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Errore: impossibile aprire la telecamera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        # Prima iterazione: crea la finestra
        ret, frame = cap.read()
        
        if not ret:
            print("Errore: impossibile leggere il primo frame")
            cap.release()
            return

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)
        
       # Ora registra il callback (la finestra è già creata)
        cv2.setMouseCallback("Camera", onMouse)
        if click_coordinates is not None:
            px, py = click_coordinates
            x_cm, y_cm = pixel_to_cm(px, py)
            print(f"Punto cliccato: ({x_cm:.2f} cm, {y_cm:.2f} cm)")
            
            # Mostra un marker sul punto cliccato
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"{x_cm:.1f}, {y_cm:.1f} cm", (px + 10, py - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Resetta per il prossimo click
            click_coordinates = None

            if px > 0:
                break
    
    return x_cm, y_cm
    cap.release()
    cv2.destroyAllWindows()
    
class DOFBOTController:
    """Controller per robot DOFBOT 6-DOF con cinematica inversa"""
    
    def __init__(self):
        # LUNGNEZZE BRACCI REALI (da tue misurazioni)
        self.l1 = 0      # Distanza base al primo giunto (asse verticale)
        self.l2 = 9.0    # Motore 2→3: 9 cm (spalla)
        self.l3 = 9.0    # Motore 3→4: 9 cm (gomito)
        self.l4 = 19.0    # Motore 4→6: 19 cm
        
        # Offset angolari (in gradi) - da calibrare per il tuo robot
        self.offsets = [0, 0, 0, 0, 0, 0]
        
        # Range di movimento per ogni motore (in gradi)
        self.angle_ranges = {
            1: (0, 180),    # Base
            2: (0, 180),    # Spalla
            3: (0, 180),    # Gomito
            4: (0, 180),    # Polso 1
            5: (0, 270),    # Polso 2
            6: (0, 180)     # Pinza
        }
        
        # Angoli attuali (in gradi)
        self.current_angles = [0, 0, 0, 0, 0, 0]
        
        # Configurazione GPIO per servomotori (se usi PWM diretto)
        if GPIO_AVAILABLE:
            self.setup_gpio()
    
    def setup_gpio(self):
        """Configura i pin GPIO per i servomotori"""
        GPIO.setmode(GPIO.BCM)
        self.servo_pins = [17, 27, 22, 23, 24, 25]
        for pin in self.servo_pins:
            GPIO.setup(pin, GPIO.OUT)
    
    def angle_to_pulse(self, angle: float, motor_num: int) -> int:
        """Converte angolo in impulso PWM (microsecondi)"""
        min_pulse = 500
        max_pulse = 2500
        max_angle = self.angle_ranges[motor_num][1]
        
        pulse = min_pulse + ((angle / max_angle) * (max_pulse - min_pulse))
        return int(pulse)
    
    def move_single_motor(self, motor_num: int, angle: float, block: bool = True):
        """Muove un singolo motore all'angolo specificato"""
        if motor_num < 1 or motor_num > 6:
            raise ValueError("Motore deve essere tra 1 e 6")
        
        min_angle, max_angle = self.angle_ranges[motor_num]
        angle = max(min_angle, min(max_angle, angle))
        angle += self.offsets[motor_num - 1]
        
        self.current_angles[motor_num - 1] = angle
        
        if GPIO_AVAILABLE:
            pulse = self.angle_to_pulse(angle, motor_num)
            print(f"Motore {motor_num}: {angle:.1f}° (pulse: {pulse}μs)")
        else:
            print(f"Motore {motor_num}: {angle:.1f}° (simulazione)")
    
    def move_all_motors(self, angles: List[float]):
        """Muove tutti i 6 motori simultaneamente"""
        if len(angles) != 6:
            raise ValueError("Devono essere forniti esattamente 6 angoli")
        
        for i, angle in enumerate(angles):
            print("move_all_motors",i)
            self.move_single_motor(i + 1, angle, block=False)
    
    def forward_kinematics(self, angles_deg: List[float]) -> Tuple[float, float, float]:
        angles_rad = [math.radians(a) for a in angles_deg]
        θ1, θ2, θ3, θ4, θ5, θ6 = angles_rad
        
        x = 0
        y = 0
        z = self.l1
        
        # Motore 1: rotazione base
        x_base = x * math.cos(θ1) - y * math.sin(θ1)
        y_base = x * math.sin(θ1) + y * math.cos(θ1)
        x, y = x_base, y_base
        
        # Motore 2-3: bracci da 9cm ciascuno (CORRETTO - senza duplicazione)
        θ23 = θ2 + θ3
        x += self.l2 * math.cos(θ2) + self.l3 * math.cos(θ23)
        z += self.l2 * math.sin(θ2) + self.l3 * math.sin(θ23)

        # Motore 4-5-6: blocco unito 19cm
        x += self.l4 * math.cos(θ23 + θ4) * math.cos(θ1)
        y += self.l4 * math.cos(θ23 + θ4) * math.sin(θ1)
        z += self.l4 * math.sin(θ23 + θ4)
        
        return (x, y, z)
        
    def cm_to_gradi(self, x: float):
        x = max(0, min(x, 6))

        punti = [
            (0, 180),
            (2, 150),
            (3.5, 120),
            (4.8, 90),
            (5.7, 60),
            (6, 0),
        ]

        for (x1, y1), (x2, y2) in zip(punti[:-1], punti[1:]):
            if x1 <= x <= x2:
                t = (x - x1) / (x2 - x1)
                return y1 + t * (y2 - y1)
    
    def inverse_kinematics(self, x: float, y: float, z: float, cm: float =6, n_it: float =200):
        # MOTORE 1: rotazione base (CORRETTO)
        cm = max(0, min(cm,6))
        
        r = math.sqrt(x**2 + y**2)
        if r == 0:
            theta_1 = 0
        else:
            theta_base = math.atan2(y, x)
            theta_1 = math.degrees(theta_base) + 90
        theta_1 = max(self.angle_ranges[1][0], min(self.angle_ranges[1][1], theta_1))
        
        # DISTANZA NEL PIANO Y-Z: usa y e z_eff
        d = math.sqrt(y**2 + x**2)  # usa y, non r!
        
        def cinematica_inversa_3link(distanza, altezza, pre):
            """
            Calcola la cinematica inversa per un robot planare a 3 link con angoli RELATIVI.
            
            Convenzione corretta per Yahboom DOFBot (motori in serie):
            - 90° = link ALLINEATO con il precedente (continua dritto)
            - < 90° = link PIEGATO verso l'interno
            - Ogni angolo è MISURATO RISPETTO AL LINK PRECEDENTE
            
            Per ottenere la posizione:
            - Angolo assoluto link 1 = 90° - theta1 (deviazione dall'orizzontale)
            - Angolo assoluto link 2 = angolo_link1 + (90° - theta2)
            - Angolo assoluto link 3 = angolo_link2 + (90° - theta3)
            
            Con theta=90°: tutti a 0° assoluti → verticale ✓
            """         
            
            x_target = distanza
            y_target = altezza
            
            portata_max =  self.l2  +  self.l3 +  self.l4
            r = np.sqrt(x_target**2 + y_target**2)
            
            if r > portata_max:
                raise ValueError(f"Distanza {r:.2f} cm oltre la portata massima di {portata_max} cm")
            
            # CONVENZIONE CORRETTA:
            # Quando theta_servo = 90°, il link è allineato col precedente
            # Deviazione dall'allineamento: delta = 90° - theta_servo
            # Angolo assoluto = somma cumulativa delle deviazioni
            
            def cinematica_directa(thetas_deg):
                theta1, theta2, theta3 = thetas_deg
                t1, t2, t3 = np.radians(theta1), np.radians(theta2), np.radians(theta3)
                
                # Deviazione da 90° (allineamento)
                delta1 = np.radians(90) - t1
                delta2 = np.radians(90) - t2
                delta3 = np.radians(90) - t3
                
                # Angolo assoluto di ogni link (rispetto all'asse orizzontale)
                # Partiamo con theta1 = 90° → verticale = 90° dall'orizzontale
                alpha1 = np.radians(90) - delta1  # = t1
                alpha2 = alpha1 - delta2           # = t1 + t2 - 90°
                alpha3 = alpha2 - delta3           # = t1 + t2 + t3 - 180°
                
                # Posizione: x = L*cos(alpha), y = L*sin(alpha)
                # Dove alpha è dall'asse orizzontale
                x =  self.l2  * np.cos(alpha1) +  self.l3  * np.cos(alpha2) +  self.l4  * np.cos(alpha3)
                y =  self.l2  * np.sin(alpha1) +  self.l3  * np.sin(alpha2) +  self.l4  * np.sin(alpha3)
                
                return np.array([x, y])
            
            def errore(thetas_deg):
                pos = cinematica_directa(thetas_deg)
                return np.sum((pos - np.array([x_target, y_target]))**2)
            
            def gradiente(theta, eps=1e-5):
                grad = np.zeros(3)
                f0 = errore(theta)
                for i in range(3):
                    theta_p = theta.copy()
                    theta_p[i] += eps
                    grad[i] = (errore(theta_p) - f0) / eps
                return grad
            
            # Punti iniziali
            punti_iniziali = [
                [80, 70, 60], [90, 90, 90], [85, 75, 65], [75, 65, 55],
                [82, 72, 62], [78, 68, 58], [88, 88, 88], [80, 80, 80]
            ]
            
            if r < 10 and y_target > 30:
                punti_iniziali = [[90, 90, 90], [88, 88, 88], [90, 87, 85]]
            elif x_target > 20 and y_target < 30:
                punti_iniziali = [[80, 70, 60], [78, 68, 58], [82, 72, 62], [77, 67, 57]]
            
            best_theta, best_err = None, float('inf')
            
            for theta_ini in punti_iniziali:
                theta = np.array(theta_ini, dtype=float)
                lr = 0.3
                #print("cinematica_inversa_3link\n")
                
                for _ in range(pre):
                    grad = gradiente(theta)
                    theta_new = np.clip(theta - lr * grad, 0, 180)
                    #print("cinematica_inversa_3link___2")
                    
                    if errore(theta_new) >= errore(theta):
                        lr *= 0.5
                        if lr < 1e-8:
                            break
                    else:
                        lr = min(lr * 1.02, 0.3)
                    theta = theta_new
                
                if errore(theta) < best_err:
                    best_err = errore(theta)
                    best_theta = theta
            
            if best_theta is None:
                raise ValueError("Impossibile trovare soluzione")
            
            return (round(np.clip(best_theta[0], 0, 180), 1),
                    round(np.clip(best_theta[1], 0, 180), 1),
                    round(np.clip(best_theta[2], 0, 180), 1))
                    
        theta_2, tetha_3, tetha_4 = cinematica_inversa_3link(distanza=d, altezza=z, pre=n_it)
        
        theta_6 = robot.cm_to_gradi(cm)
        
        return (theta_1, theta_2, tetha_3, tetha_4, 90, theta_6)

    
    def move_to_position(self, x: float, y: float, z: float, cm: float =6, n_it: float =200):
        """Muove il braccio alla posizione (x, y, z) usando cinematica inversa"""
        angles = self.inverse_kinematics(x, y, z, cm, n_it)
        
        print(f"Target: ({x:.2f}, {y:.2f}, {z:.2f}) cm")
        print(f"Angoli calcolati: {[f'{a:.1f}°' for a in angles]}")
        
        return list(angles)
    
    def calibrate_offsets(self, offsets: List[float]):
        """Imposta offset di calibrazione per tutti i motori"""
        if len(offsets) != 6:
            raise ValueError("Devono essere forniti esattamente 6 offset")
        self.offsets = offsets
    
    def close(self):
        """Pulizia GPIO"""
        if GPIO_AVAILABLE:
            GPIO.cleanup()


if __name__ == "__main__":
    # Crea controller
    robot = DOFBOTController()
    
    try:
        import time
        time.sleep(0.2)
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        while True:
            # muove il braccio in posizione home
            arm.Arm_serial_servo_write6(90, 90, 0, 0, 90, 0, 5000)
            time.sleep(5)
            
            # muove il braccio sul oggetto da prendere
            x, y = centrale()
            angles = robot.move_to_position(-y + 14.5, -x, z=-11, cm = 6)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 3000)
            time.sleep(3)
            
            # prende l'oggetto
            apertura = 1.8
            ap = robot.cm_to_gradi(apertura)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], ap, 3000)
            time.sleep(3)
            # si alza
            angles = robot.move_to_position(-y + 14.5, -x, z=5, cm=apertura, n_it=10)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
            time.sleep(2)
            
            # muove il braccio alla posizione di place
            angles = robot.move_to_position(x=10, y=15, z=-10, cm=apertura, n_it=50)
            arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 1300)
            time.sleep(1.3)
            
            # molla l'oggetto
            arm.Arm_serial_servo_write6(angles[0], 40, 45, 17, 90, 0, 1000)
            time.sleep(1)
            
        
    finally:
        robot.close()
