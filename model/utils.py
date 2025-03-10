import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def compare_masks(mask, true_mask):
    _, ground_truth = cv2.threshold(true_mask, 127, 255, cv2.THRESH_BINARY)
    _, reconstructed = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    height, width = ground_truth.shape
    visualization = np.zeros((height, width, 3), dtype=np.uint8)

    true_positives = (ground_truth == 255) & (reconstructed == 255)
    false_positives = (ground_truth == 0) & (reconstructed == 255)
    false_negatives = (ground_truth == 255) & (reconstructed == 0)

    visualization[true_positives] = [0, 255, 0]  # Зелёный для верно восстановленных
    visualization[false_positives] = [0, 0, 255]  # Красный для ложноположительных
    visualization[false_negatives] = [255, 0, 0]  # Синий для ложноотрицательных
    return visualization


def _get_inception_model():
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model


def _get_activations(images, model, batch_size=2, dims=2048):
    model.eval()
    activations = np.empty((len(images), dims))
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            pred = model(batch)
            activations[i:i + batch_size] = pred.cpu().numpy()
    return activations


def _calculate_FID(real_activations, fake_activations):
    mu1, sigma1 = real_activations.mean(axis=0), np.cov(real_activations, rowvar=False)
    mu2, sigma2 = fake_activations.mean(axis=0), np.cov(fake_activations, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def FID(real_images, fake_images):
    inception_model = _get_inception_model().to(device)

    real_activations = _get_activations(real_images, inception_model)
    fake_activations = _get_activations(fake_images, inception_model)

    return _calculate_FID(real_activations, fake_activations)


def SLM(base_img, new_img, mask):
    return mask * new_img + (1 - mask) * base_img


if __name__ == "__main__":
    # Тестирование сравнения масок
    '''mask_true = cv2.imread('C:/Users/phone/2D_bar_codes/test_utils/mask_true.png', cv2.IMREAD_GRAYSCALE)
    mask_rec1 = cv2.imread('C:/Users/phone/2D_bar_codes/test_utils/mask_rec_1.png', cv2.IMREAD_GRAYSCALE)
    mask_rec2 = cv2.imread('C:/Users/phone/2D_bar_codes/test_utils/mask_rec_2.png', cv2.IMREAD_GRAYSCALE)
    compare1 = compare_masks(mask_rec1, mask_true)
    compare2 = compare_masks(mask_rec2, mask_true)
    cv2.imwrite('C:/Users/phone/2D_bar_codes/test_utils/compare1.png', compare1)
    cv2.imwrite('C:/Users/phone/2D_bar_codes/test_utils/compare2.png', compare2)'''

    '''# Тестирование FID
    transform = Compose([
        Resize((299, 299)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    syn_names = ["5.png", "6.png", "0020.png", "0021.png", "0025.png", "0026.png", "0598.png", "0642.png"]
    aug_names = [f"{i}.png" for i in range(12, 20)]
    real_names = [f"{i}.png" for i in range(7, 15)]
    path = "C:/Users/phone/2D_bar_codes/test_utils/i_s/"
    syn_img, aug_img, real_img = [], [], []
    for name in syn_names:
        img = Image.open(path + name).convert('RGB')
        syn_img.append(transform(img))
    for name in aug_names:
        img = Image.open(path + name).convert('RGB')
        aug_img.append(transform(img))
    for name in real_names:
        img = Image.open(path + name).convert('RGB')
        real_img.append(transform(img))
    dist1 = FID(torch.stack(real_img), torch.stack(aug_img))
    dist2 = FID(torch.stack(real_img), torch.stack(syn_img))
    print(f"FID для аугментированных и реальных: {dist1}")
    print(f"FID для синтетических и реальных: {dist2}")'''

    '''mask = cv2.imread('C:/Users/phone/MIPT_conf_2025/1_o_mask_t.png')
    base = cv2.imread('C:/Users/phone/MIPT_conf_2025/1_o_b.png')
    new = cv2.imread('C:/Users/phone/MIPT_conf_2025/1_o_f.png')
    mask = mask > 0

    slm_img = SLM(base, new, mask)
    cv2.imwrite('C:/Users/phone/MIPT_conf_2025/1_slm.png', slm_img)'''

    for i in range(20, 21):
        mask = cv2.imread(f'C:/Users/phone/2D_bar_codes/model/output/full-train/val_visualization/iter-40000/{i}_o_mask_t.png')
        base = cv2.imread(f'C:/Users/phone/2D_bar_codes/model/output/full-train/val_visualization/iter-40000/{i}_o_b.png')
        new = cv2.imread(f'C:/Users/phone/2D_bar_codes/model/output/full-train/val_visualization/iter-40000/{i}_o_f.png')
        mask = mask > 0

        slm_img = SLM(base, new, mask)
        cv2.imwrite(f'C:/Users/phone/2D_bar_codes/model/output/full-train/val_visualization/iter-40000/{i}_slm.png', slm_img)
