import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
from model import BiSeNet

def vis_parsing_maps(im, parsing_anno, stride=1):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255


    # 17은 헤어 클래스 인덱스입니다
    hair_index = np.where((vis_parsing_anno == 17) | (vis_parsing_anno == 2))
    vis_parsing_anno_color[hair_index[0], hair_index[1], :] = [255, 0, 0]  # 빨간색으로 헤어 표시

    # num_of_class = np.max(vis_parsing_anno)
    # for pi in range(1, num_of_class + 1):
    #     index = np.where(vis_parsing_anno == pi)
    #     vis_parsing_anno_color[index[0], index[1], :] = [255, 0, 0]  # 빨간색으로 헤어 표시

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    return vis_im

def hair_segmentation(image_path):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cpu()
    net.load_state_dict(torch.load('79999_iter.pth', map_location='cpu'))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = Image.open(image_path)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cpu()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    vis_im = vis_parsing_maps(image, parsing)
    cv2.imwrite('result.jpg', vis_im)
    print('Result saved as result.jpg')

if __name__ == '__main__':
    hair_segmentation('image/image3.png')