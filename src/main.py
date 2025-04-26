import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from models.superpoint import SuperPointNet
from models.superglue import SuperGlueNet

def carregar_imagem(caminho):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"⚠️ Não foi possível carregar a imagem: {caminho}")
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return img

def detectar_keypoints(modelo_superpoint, imagem):
    modelo_superpoint.eval()
    with torch.no_grad():
        det, desc = modelo_superpoint(imagem)
        keypoints = torch.nonzero(det.squeeze(0).squeeze(0) > 0.2, as_tuple=False)  # threshold 0.2
        descritores = desc.squeeze(0).permute(1, 2, 0)[keypoints[:, 0], keypoints[:, 1]]
    return keypoints, descritores

def visualizar_matches(img1_np, img2_np, kp1, kp2, matches_idx):
    h1, w1 = img1_np.shape
    h2, w2 = img2_np.shape

    img_out = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    img_out[:h1, :w1] = img1_np
    img_out[:h2, w1:] = img2_np

    plt.figure(figsize=(15, 8))
    plt.imshow(img_out, cmap='gray')
    for idx1, idx2 in matches_idx:
        pt1 = (kp1[idx1][1].item(), kp1[idx1][0].item())
        pt2 = (kp2[idx2][1].item() + w1, kp2[idx2][0].item())
        plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], 'r-', linewidth=0.5)
    plt.axis('off')
    plt.show()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")

    # Instanciar modelos
    superpoint = SuperPointNet().to(device)
    superglue = SuperGlueNet().to(device)

    # Pasta das imagens
    assets_path = os.path.join(os.getcwd(), 'assets')
    imagens = [f for f in os.listdir(assets_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if len(imagens) < 2:
        print("⚠️ São necessárias pelo menos duas imagens na pasta assets/.")
        return

    # Carregar imagens
    imagens.sort()  # organiza ordem alfabética
    imgs_torch = []
    imgs_np = []
    for img_name in imagens:
        img_path = os.path.join(assets_path, img_name)
        imgs_torch.append(carregar_imagem(img_path).to(device))
        imgs_np.append(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
    
    # Detectar keypoints e descritores
    keypoints_list = []
    descritores_list = []
    for img in imgs_torch:
        kp, desc = detectar_keypoints(superpoint, img)
        keypoints_list.append(kp)
        descritores_list.append(desc)

    # Matching entre pares de imagens
    superglue.eval()
    for i in range(len(imgs_torch)):
        for j in range(i + 1, len(imgs_torch)):
            print(f"🔗 Matching: {imagens[i]} <--> {imagens[j]}")

            kp1 = keypoints_list[i]
            desc1 = descritores_list[i]
            kp2 = keypoints_list[j]
            desc2 = descritores_list[j]

            if kp1.shape[0] == 0 or kp2.shape[0] == 0:
                print(f"⚠️ Sem keypoints suficientes entre {imagens[i]} e {imagens[j]}.")
                continue

            with torch.no_grad():
                scores = superglue(desc1.unsqueeze(0), desc2.unsqueeze(0))
                scores = scores.squeeze(0).squeeze(-1)

            matches_idx = torch.argmax(scores, dim=1)
            matches = [(idx1, idx2.item()) for idx1, idx2 in enumerate(matches_idx)]

            visualizar_matches(imgs_np[i], imgs_np[j], kp1, kp2, matches)

if __name__ == "__main__":
    main()
