import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("alarm.mp3")
pygame.mixer.music.play()

print("Playing sound...")
time.sleep(5)

pygame.mixer.music.stop()
