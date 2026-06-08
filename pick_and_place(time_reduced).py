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
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO non disponibile (non su Raspberry Pi)")

import numpy as np
from typing import Tuple, List


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
    
    def inverse_kinematics(self, x: float, y: float, z: float, cm: float):
        # MOTORE 1: rotazione base (CORRETTO)
        cm = max(0, min(cm,6))
        
        r = math.sqrt(x**2 + y**2)
        if r == 0:
            theta_1 = 0
        else:
            theta_base = math.atan2(y, x)
            theta_1 = math.degrees(theta_base) + 90
        theta_1 = max(self.angle_ranges[1][0], min(self.angle_ranges[1][1], theta_1))
        
        # DISTANZA NEL PIANO Y-Z: usa y (non r!) e z_eff
        d = math.sqrt(y**2 + x**2)  # CORREZIONE: usa y, non r!
        
        def cinematica_inversa_3link(distanza, altezza):
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
            L1 = 9.0
            L2 = 9.0
            L3 = 19.0
            
            x_target = distanza
            y_target = altezza
            
            portata_max = L1 + L2 + L3
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
                x = L1 * np.cos(alpha1) + L2 * np.cos(alpha2) + L3 * np.cos(alpha3)
                y = L1 * np.sin(alpha1) + L2 * np.sin(alpha2) + L3 * np.sin(alpha3)
                
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
                
                for _ in range(3000):
                    grad = gradiente(theta)
                    theta_new = np.clip(theta - lr * grad, 0, 180)
                    
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
                    
        theta_2, tetha_3, tetha_4 = cinematica_inversa_3link(distanza=d, altezza=z)
        
        #calcola theta_6 l'apertura della pinza in cm
        #theta_6 = 180 - (cm*180/6)
        
        def cm_to_gradi(x=cm):
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
       
        theta_6 = cm_to_gradi(cm)
        return (theta_1, theta_2, tetha_3, tetha_4, 90, theta_6)

    
    def move_to_position(self, x: float, y: float, z: float, cm: float):
        """Muove il braccio alla posizione (x, y, z) usando cinematica inversa"""
        angles = self.inverse_kinematics(x, y, z, cm)
        
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
        
        # porta il braccio alla posizione home
        angles = robot.move_to_position(x=0, y=10, z=15, cm=6)
        print("\nporta il braccio alla posizione home")
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
        
        time.sleep(0.1)
        
        # muove il braccio in una posizione random al interno di un area, la posizione di pick
        print("\nmuove il braccio in una posizione random al interno di un area, la posizione di pick")
        
        x=random.uniform(0,15)
        y=random.uniform(10,25)
        angles = robot.move_to_position(x, y, z=-7, cm=6)
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
        
        time.sleep(0.2)
        
        
        # prende l'oggetto
        print("\nprende l'oggetto")
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], -10, 90, 180, 2000)
        time.sleep(0.2)
        print("\nSi alza")
        angles = robot.move_to_position(x, y, z=5, cm=0)
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
        time.sleep(0.2)
        
        
        # muove il braccio alla posizione di place
        print("\nmuove il braccio alla posizione di place")
        angles = robot.move_to_position(x=20, y=-10, z=-10, cm=0)
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
        
        time.sleep(0.5)
        
        # molla l'oggetto
        print("\nmolla l'oggetto")
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        
        arm.Arm_serial_servo_write6(angles[0], 40, 45, 17, 90, 0, 2000)
        time.sleep(0.2)
        
        
        
        # porta il braccio alla posizione home
        print("porta il braccio alla posizione home")
        angles = robot.move_to_position(x=0, y=10, z=15, cm=6)
        
        from Arm_Lib import Arm_Device
        arm = Arm_Device()
        time.sleep(0.1)
        
        arm.Arm_serial_servo_write6(angles[0], angles[1], angles[2], angles[3], angles[4], angles[5], 2000)
        
        time.sleep(0.2)
        
    finally:
        robot.close()