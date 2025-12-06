from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from os.path import join
from os import listdir
import cv2
import numpy as np
import hdf5storage as h5 
from diffusers import DDPMPipeline

def load_image(filename):
    return Image.open(filename).convert('RGB')

def wc_norm(wc_path):
    wc=cv2.imread(wc_path,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # scale wc
    xmx, xmn, ymx, ymn,zmx, zmn= 1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497     ## value obtained from the entire dataset 
    wc[:,:,0]= (wc[:,:,0]-zmn)/(zmx-zmn)
    wc[:,:,1]= (wc[:,:,1]-ymn)/(ymx-ymn)
    wc[:,:,2]= (wc[:,:,2]-xmn)/(xmx-xmn)

    return wc

class Doc3d(Dataset):
    def __init__(self, root_dir, args, mask = False, resize=False):
        img_dir = join(root_dir, 'img')
        bm_dir = join(root_dir, 'bm') 
        wc_dir = join(root_dir, 'wc')
        recon_dir = join(root_dir, 'recon')
        folders = listdir(img_dir)
        self.mask = mask
        self.resize = resize
        self.args = args

        self.all_imgs, self.all_bms, self.all_recons, self.all_wcs = [], [], [], []
        for folder in folders:
            img_folder_path = join(img_dir, folder)
            img_paths = [join(img_folder_path, i) for i in listdir(img_folder_path)]
            self.all_imgs.extend(img_paths)

            wc_paths, bm_paths, recon_paths = [], [], []
            for img_path in img_paths:
                name = img_path.split('/')[-1].split('.')[0]
                bm_folder_path = join(bm_dir, folder)
                wc_folder_path = join(wc_dir, folder)
                bm_paths.append(join(bm_folder_path, name + '.mat'))
                wc_paths.append(join(wc_folder_path, name + '.exr'))

                recon_folder_path = join(recon_dir, folder)
                recon_paths.append(join(recon_folder_path, name[:-4]+'chess480001.png'))
            self.all_recons.extend(recon_paths)
            self.all_bms.extend(bm_paths)
            self.all_wcs.extend(wc_paths)

    def extract_textline(self, gt, idx=None):
        gt = (gt*255).to(torch.uint8).numpy()
        gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)[10:-10, 10:-10]
        intensity = np.mean(gray)
        global count 
        if intensity > 50:
            bi_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        #_, bi_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            count.append(idx)
            bi_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        di_img = cv2.dilate(bi_img, kernel, iterations=1)

        contours, _ = cv2.findContours(di_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 25 and h < 20:
                filtered_contours.append(contour)

        fil_img = np.zeros_like(bi_img)
        cv2.drawContours(fil_img, filtered_contours, -1, (255), thickness=cv2.FILLED)

        textlines = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            horizontal_length = w
            textlines.append((center_x, center_y, horizontal_length))

        # Draw the textlines on the original image

        #textline_img = np.ascontiguousarray(gt, dtype=np.uint8)
        textline_img = np.ascontiguousarray(np.zeros_like(gt, dtype=np.uint8))
        
        for center_x, center_y, horizontal_length in textlines:
            cv2.line(textline_img, (center_x+10 - horizontal_length // 2, center_y+10), 
                    (center_x+10 + horizontal_length // 2, center_y+10), (255, 255, 255), 2)

        textline_img = cv2.cvtColor(textline_img, cv2.COLOR_BGR2GRAY)
        
        #cv2.imwrite(f'tmp/{idx}_gray.png', gray)
        #cv2.imwrite(f'tmp/{idx}_binary.png', bi_img)
        #cv2.imwrite(f'tmp/{idx}_dilate.png', di_img)
        #cv2.imwrite(f'tmp/{idx}_filtered.png', fil_img)
        #cv2.imwrite(f'tmp/{idx}_textline.png', textline_img)

        return torch.from_numpy(textline_img).unsqueeze(0)
        #return torch.from_numpy(textline_img).permute(2, 0, 1)

    def reverse_bm(self, gt, bm):
        int_bm = (gt.shape[-1] * (bm/2 + 0.5)).to(torch.long)
        recon_img = torch.zeros_like(gt)
        y, x = int_bm[..., 0], int_bm[..., 1]
        recon_img[..., x, y] = gt
        
        return recon_img

    def __len__(self):
        assert (len(self.all_bms) == len(self.all_imgs)) and (len(self.all_recons) == len(self.all_imgs))
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = self.all_imgs[idx]
        bm_path = self.all_bms[idx]
        recon_path = self.all_recons[idx]
        wc_path = self.all_wcs[idx]

        img = load_image(img_path)
        img = np.asarray(img)/255

        bm = h5.loadmat(bm_path)['bm']
        bm /= np.array([448, 448])
        wc = wc_norm(wc_path)

        if self.mask:
            recon = np.asarray(load_image(recon_path))
            recon = np.sum(recon, axis=-1)[..., np.newaxis].astype(bool).astype(np.uint8)
            img *= recon

        img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous()
        bm = torch.from_numpy((bm - 0.5) * 2).float().contiguous()
        
        if self.resize:
            img = img.permute(1, 2, 0).numpy()
            bm = bm.numpy()

            img = cv2.resize(img, (self.args.image_size, self.args.image_size))
            bm0 = cv2.resize(bm[..., 0], (self.args.image_size, self.args.image_size))
            bm1 = cv2.resize(bm[..., 1], (self.args.image_size, self.args.image_size))
            wc0 = cv2.resize(wc[..., 0], (self.args.image_size, self.args.image_size))
            wc1 = cv2.resize(wc[..., 0], (self.args.image_size, self.args.image_size))
            wc2 = cv2.resize(wc[..., 0], (self.args.image_size, self.args.image_size))
            
            wc = np.stack([wc0, wc1, wc2], axis=-1)
            bm = np.stack([bm0, bm1], axis=-1)
        
        
        img = torch.from_numpy(img).permute(2, 0, 1).float().contiguous() if not isinstance(img, torch.Tensor) else img
        bm = torch.from_numpy(bm).float().contiguous() if not isinstance(bm, torch.Tensor) else bm
        wc = torch.from_numpy(wc).float().contiguous().permute(2, 0, 1) if not isinstance(wc, torch.Tensor) else wc
        
        return {
            'img': 2*img - 1,
            'bm': bm.permute(2, 0, 1),
            'wc': wc
        }
        
        
class Data(Dataset):
    def __init__(self, root, transform=None):
        ys_root = os.path.join(root, 'digital')
        xs_root = os.path.join(root, 'image')
        
        ys_subfolders = [os.path.join(ys_root, i) for i in os.listdir(ys_root)]
        xs_subfolders = [os.path.join(xs_root, i) for i in os.listdir(xs_root)]
        xs, ys = [], []
        for i in xs_subfolders:
            xs.extend(
                [os.path.join(i, j) for j in os.listdir(i)]
            )
        
        for i in ys_subfolders:
            ys.extend(
                [os.path.join(i, j) for j in os.listdir(i)]
            )
        
        self.xs = sorted(xs)
        self.ys = sorted(ys)      
        
        self.transform = transform  


    def __len__(self):
        assert len(self.xs) == len(self.ys), "Number of images must be the same."
        return len(self.xs)
    
    def __getitem__(self, idx):
        x = Image.open(self.xs[idx])
        y = Image.open(self.ys[idx])
        
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        
        return x, y
    
if __name__ == "__main__":
    root = './data/uds-rift-cat-munchkin-250914'
    data = Doc3d(root)[0]
    img = data['img'].unsqueeze(0)
    bm = data['bm'].unsqueeze(0)
    
    recon = grid_sample(img, bm).numpy()[0]
    image_data = (np.transpose(recon, (1, 2, 0))*255).astype(np.uint8)
    image = Image.fromarray(image_data)
    image.save('./tmp.png')
