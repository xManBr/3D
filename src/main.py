import torch
import torch.nn.functional as F
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
        semi, desc = modelo_superpoint(imagem)

        heatmap = F.softmax(semi, dim=1)[:, :-1, :, :]  # Ignora dustbin
        heatmap = heatmap.squeeze(0)

        num_keypoints = 500

        heatmap_flat = heatmap.view(heatmap.shape[0], -1)
        scores, indices = torch.topk(heatmap_flat, num_keypoints)

        keypoints = []
        Hc, Wc = heatmap.shape[1], heatmap.shape[2]
        for i in range(indices.shape[1]):  # <- Correção aqui!
            idx = indices[0, i]
            y = idx // Wc
            x = idx % Wc
            keypoints.append([y.item() * 8, x.item() * 8])  # Corrige escala

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        desc = F.normalize(desc, p=2, dim=1)
        desc = desc.squeeze(0).permute(1, 2, 0)

        descritores = []
        for pt in keypoints:
            y, x = int(pt[0] / 8), int(pt[1] / 8)
            if y >= desc.shape[0] or x >= desc.shape[1]:
                continue
            descritores.append(desc[y, x])

        descritores = torch.stack(descritores)

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
    print(os.getcwd())
    # 👉 Adicionar caminho para pesos
    weights_path = os.path.join(os.getcwd(),'src', 'models', 'weights', 'superpoint_v1.pth')

    print(weights_path)
    # Instanciar modelos já carregando pesos
    superpoint = SuperPointNet(weights_path=weights_path).to(device)
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

            #visualizar_matches(imgs_np[i], imgs_np[j], kp1, kp2, matches)
    
    matches_colmap = []

    # Para cada par de imagens
    for i in range(len(imgs_torch)):
        for j in range(i + 1, len(imgs_torch)):
            print(f"🔗 Matching: {imagens[i]} <--> {imagens[j]}")
            
            # Calcular scores SuperGlue
            scores = superglue(descritores_list[i].unsqueeze(0), descritores_list[j].unsqueeze(0))
            scores = scores.squeeze(0)
            
            matches_idx = torch.argmax(scores, dim=1)
            matches_score = torch.max(scores, dim=1).values

            threshold = 0.7  # Score mínimo
            matches = []
            for idx1, (idx2, score) in enumerate(zip(matches_idx, matches_score)):
                if score.item() > threshold:
                    matches.append((idx1, idx2.item()))

            if matches:
                matches_colmap.append((imagens[i], imagens[j], matches))

    # Agora salvar no formato que o COLMAP entende
    with open('matches/matches.txt', 'w') as f:
        for img1, img2, matches in matches_colmap:
            f.write(f"{img1} {img2}\n")
            for m in matches:
                f.write(f"{m[0]} {m[1]}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
