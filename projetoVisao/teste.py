from ultralytics import YOLO 
import cv2 
# --- CONFIGURAÇÃO --- # Em vez de 'yolov8n.pt', carregamos O NOSSO ARQUIVO! 

nome_modelo = r'C:\Users\davis\OneDrive\Documentos\Faculdade\Arquivo Python\visaoComputacional\runs\detect\train-2\weights\best.pt' 
# Fonte de vídeo (0 para webcam, ou nome de um arquivo mp4) 

source = r'projetoVisao\video_test\teste1.mp4'
model = YOLO(nome_modelo) # Verificar quais classes esse modelo conhece 
print("Classes que este modelo conhece:", model.names) # Rodar inferência no vídeo (show=True abre a janelinha) # conf=0.5 significa que só mostra se tiver 50% de certeza 
model.predict(source=source, show=True, conf=0.6)